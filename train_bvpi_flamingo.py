import argparse
import json
import os
import torch

os.environ["ATEN_NNPACK_ENABLED"] = "0"

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, PreTrainedModel, get_constant_schedule_with_warmup
from torch.optim import AdamW
from open_flamingo import create_model_and_transforms

# ============================================================
# 🔧 PATCHES FOR MPT / MOSAIC GPT
# ============================================================

def patch_mosaic_mpt():
    orig_get = PreTrainedModel.get_input_embeddings
    orig_set = PreTrainedModel.set_input_embeddings

    def get_input_embeddings(self):
        if hasattr(self, "transformer") and hasattr(self.transformer, "wte"):
            return self.transformer.wte
        return orig_get(self)

    def set_input_embeddings(self, value):
        if hasattr(self, "transformer") and hasattr(self.transformer, "wte"):
            self.transformer.wte = value
            return
        return orig_set(self, value)

    PreTrainedModel.get_input_embeddings = get_input_embeddings
    PreTrainedModel.set_input_embeddings = set_input_embeddings


def patch_mosaic_config():
    cfg = AutoConfig.from_pretrained(
        "anas-awadalla/mpt-1b-redpajama-200b",
        trust_remote_code=True
    )
    if not hasattr(cfg.__class__, "hidden_size"):
        cfg.__class__.hidden_size = property(lambda self: self.d_model)


patch_mosaic_mpt()
patch_mosaic_config()

# ============================================================
# 📦 DATASET
# ============================================================

class VLMDataset(Dataset):
    def __init__(self, path, image_root, image_processor, tokenizer,
                 max_length=256, text_dropout=0.3):
        self.image_root = image_root
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_dropout = text_dropout

        self.data = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        self.is_prompted = "prompt" in self.data[0]
        print(f"Loaded {len(self.data)} examples | "
              f"mode={'prompted' if self.is_prompted else 'caption'} | "
              f"text_dropout={text_dropout}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        try:
            image = Image.open(
                os.path.join(self.image_root, item["image"])
            ).convert("RGB")
        except Exception as e:
            print(f"Image load error ({item['image']}): {e} — using blank")
            image = Image.new("RGB", (224, 224), 0)

        vision_x = self.image_processor(image)
        vision_x = vision_x.unsqueeze(0).unsqueeze(0)   # [1, 1, C, H, W]

        if self.is_prompted:
            prompt   = item["prompt"].strip()
            response = item["response"].strip()

            # Text dropout: randomly remove structured field labels so the
            # model cannot shortcut by memorising label→value patterns.
            # It must instead read the image to fill in the values.
            if torch.rand(1).item() < self.text_dropout:
                for key in ["Sidewalk:", "Road:", "RoadVsSidewalk:", "Nearby:", "Far:"]:
                    response = response.replace(key, "")

            text        = f"<image>User: {prompt}\nAssistant: {response}<|endofchunk|>"
            prompt_text = f"<image>User: {prompt}\nAssistant: "
        else:
            text        = f"<image>{item['caption']}<|endofchunk|>"
            prompt_text = "<image>"

        tok = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = tok.input_ids.clone()
        # Mask padding
        labels[labels == self.tokenizer.pad_token_id] = -100
        # Mask prompt — only supervise the response tokens
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids
        prompt_len = min(prompt_ids.shape[1], labels.shape[1])
        labels[:, :prompt_len] = -100

        return {
            "vision_x":      vision_x,
            "input_ids":     tok.input_ids.squeeze(0),
            "attention_mask": tok.attention_mask.squeeze(0),
            "labels":        labels.squeeze(0),
        }


# ============================================================
# 🧠 TRAINING
# ============================================================

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
        cross_attn_every_n_layers=1,
        decoder_layers_attr_name="transformer.blocks",
    )

    if hasattr(model.lang_encoder, "transformer") and \
       hasattr(model.lang_encoder.transformer, "blocks"):
        print("Aliasing model.lang_encoder.layers -> transformer.blocks (MPT fix)")
        model.lang_encoder.layers = model.lang_encoder.transformer.blocks

    tokenizer.pad_token = tokenizer.eos_token

    # ── Freeze everything except gated cross-attention ──────────────────
    for name, p in model.named_parameters():
        if "gated_cross_attn" not in name:
            p.requires_grad = False
    # Belt-and-suspenders: also freeze non-cross-attn lang encoder params
    for name, p in model.named_parameters():
        if "lang_encoder" in name and "gated_cross_attn" not in name:
            p.requires_grad = False

    # ── FIX 1: Gate init ONLY when starting from scratch ────────────────
    # When resuming, the checkpoint already carries the learned gate values
    # (0.63+). Running init here would overwrite them with 0.5 before the
    # checkpoint loads — or after if load order is wrong in a wrapper.
    # Guard against both cases by skipping init entirely on resume.
    if not args.resume_from:
        print(f"Fresh start — initialising gates to {args.gate_init}")
        for name, p in model.named_parameters():
            if "gated_cross_attn" in name and "attn_gate" in name:
                torch.nn.init.constant_(p.data, args.gate_init)
    else:
        print("Resuming — skipping gate init (checkpoint values will be used)")

    # ── FIX 2: Load checkpoint HERE, before optimizer setup ─────────────
    # Previously loading happened in an external wrapper after train() set
    # up the optimizer, meaning gate_init could race with the load.
    # Loading inside train() gives a guaranteed correct order:
    #   create model → (optionally) init gates → load checkpoint → optimizer
    if args.resume_from:
        print(f"🔁 Loading checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"Checkpoint loaded. "
              f"Missing keys: {len(missing)}  Unexpected keys: {len(unexpected)}")

    model.to(device).train()

    # Log gate state immediately after load so we can verify
    gate_vals_start = [
        p.mean().item() for n, p in model.named_parameters() if "attn_gate" in n
    ]
    if gate_vals_start:
        print(f"Gates at training start: "
              f"mean={sum(gate_vals_start)/len(gate_vals_start):.4f}  "
              f"min={min(gate_vals_start):.4f}  "
              f"max={max(gate_vals_start):.4f}")

    # ── Dataset & loader ────────────────────────────────────────────────
    train_ds = VLMDataset(
        args.dataset,
        args.image_root,
        image_processor,
        tokenizer,
        max_length=args.max_length,
        text_dropout=args.text_dropout,
    )
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    # ── FIX 3: gate_lr_multiplier = 0.0 freezes gates in optimizer ──────
    # Gates are already open at 0.65. Applying 50× LR to saturated gates:
    #   - Contributes nothing useful (gates are where we want them)
    #   - Creates effective gate LR of 3e-5 × 50 = 1.5e-3, which is the
    #     cause of the loss spikes seen at steps 150-200 in earlier runs
    # Solution: exclude gate params from the optimizer entirely when
    # gate_lr_multiplier == 0. This is the cleanest freeze mechanism.
    other_params = []
    gate_params  = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "attn_gate" in n:
            gate_params.append(p)
        else:
            other_params.append(p)

    if args.gate_lr_multiplier > 0.0:
        effective_gate_lr = args.lr * args.gate_lr_multiplier
        print(f"Gates in optimizer | gate LR = {effective_gate_lr:.2e}")
        param_groups = [
            {"params": other_params, "lr": args.lr,           "name": "cross_attn"},
            {"params": gate_params,  "lr": effective_gate_lr, "name": "gates"},
        ]
    else:
        print("Gates FROZEN (gate_lr_multiplier=0.0) — only Q/K/V weights update")
        for p in gate_params:
            p.requires_grad = False   # belt-and-suspenders: exclude from autograd too
        param_groups = [{"params": other_params, "lr": args.lr, "name": "cross_attn"}]

    trainable = sum(p.numel() for p in other_params)
    gate_count = sum(p.numel() for p in gate_params)
    print(f"Trainable params: {trainable:,}  |  Gate params: {gate_count:,} "
          f"({'frozen' if args.gate_lr_multiplier == 0 else 'trainable'})")

    optim = AdamW(param_groups, weight_decay=0.01)
    sched = get_constant_schedule_with_warmup(
        optim,
        num_warmup_steps=max(100, len(loader)),
    )

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    best_loss = float("inf")

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        epoch_losses = []

        for step, batch in enumerate(loader):
            optim.zero_grad()

            out = model(
                vision_x=batch["vision_x"].to(device),
                lang_x=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device).bool(),
            )

            logits = out.logits[..., :-1, :]
            labels = batch["labels"].to(device)[..., 1:]

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

            # ── FIX 4: Gate regularizer direction reversed ───────────────
            # OLD (wrong):  (0.6 - p.mean()).clamp(min=0)
            #   → zero gradient when gates > 0.6 (which they are).
            #   → did absolutely nothing with gates at 0.653.
            #
            # NEW (correct): (p.mean() - 0.5).clamp(min=0)
            #   → fires when gates > 0.5, applying downward pressure.
            #   → with gates at 0.653: penalty = 0.153 (active) ✓
            #   → gently prevents further creep toward 1.0
            #
            # Only apply when gates are actually in the optimizer;
            # applying when frozen wastes a forward pass.
            gate_penalty = torch.tensor(0.0, device=device)
            if args.gate_reg > 0.0 and args.gate_lr_multiplier > 0.0:
                for n, p in model.named_parameters():
                    if "attn_gate" in n and p.requires_grad:
                        gate_penalty = gate_penalty + (p.mean() - 0.5).clamp(min=0)

            total_loss = loss + args.gate_reg * gate_penalty
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            epoch_losses.append(loss.item())

            if step % 50 == 0:
                gate_vals = [
                    p.mean().item() for n, p in model.named_parameters()
                    if "attn_gate" in n
                ]
                avg_gate = sum(gate_vals) / len(gate_vals) if gate_vals else 0.0
                # Print 4 decimal places so tiny gate drift is visible
                print(
                    f"Epoch {epoch+1} Step {step:>4} | "
                    f"Loss {loss.item():.4f} | "
                    f"Avg Gate {avg_gate:.4f}"
                )

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\n{'='*55}")
        print(f"Epoch {epoch+1} complete | Avg Loss: {avg_epoch_loss:.4f}")
        print(f"{'='*55}\n")

        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_path = f"{args.output_dir}/epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved → {ckpt_path}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = f"{args.output_dir}/best_model.pt"
            torch.save(model.state_dict(), best_path)
            print(f"✓ New best model (loss={best_loss:.4f}) → {best_path}")


# ============================================================
# 🚀 CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_root",  type=str, required=True)
    parser.add_argument("--dataset",     type=str, required=True)
    parser.add_argument("--output_dir",  type=str, default="checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint .pt to resume from")

    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch_size", type=int,   default=2)
    # FIX: default LR lowered from 3e-5 → 1e-5 for warm checkpoint resuming
    parser.add_argument("--lr",         type=float, default=1e-5)
    # FIX: default multiplier changed from 50 → 0 (freeze gates on resume)
    parser.add_argument("--gate_lr_multiplier", type=float, default=0.0,
                        help="Set >0 only when training gates from scratch. "
                             "Use 0.0 (default) when resuming — gates already open.")

    parser.add_argument("--gate_init", type=float, default=0.5,
                        help="Initial gate value for fresh (non-resume) runs only.")
    # FIX: gate_reg now pulls gates DOWN (toward 0.5), not up (toward 0.6)
    parser.add_argument("--gate_reg",     type=float, default=0.10,
                        help="Penalty weight for gates exceeding 0.5. "
                             "Irrelevant when gate_lr_multiplier=0.")
    parser.add_argument("--text_dropout", type=float, default=0.25)
    parser.add_argument("--max_length",   type=int,   default=448)

    args = parser.parse_args()

    print("=" * 55)
    print("Training configuration")
    print("=" * 55)
    print(f"  resume_from        : {args.resume_from or 'None (fresh start)'}")
    print(f"  lr                 : {args.lr}")
    print(f"  gate_lr_multiplier : {args.gate_lr_multiplier} "
          f"({'FROZEN' if args.gate_lr_multiplier == 0 else 'ACTIVE'})")
    print(f"  gate_reg           : {args.gate_reg}")
    print(f"  gate_init          : {args.gate_init} "
          f"({'skipped — resuming' if args.resume_from else 'will apply'})")
    print(f"  text_dropout       : {args.text_dropout}")
    print(f"  epochs             : {args.epochs}")
    print(f"  max_length         : {args.max_length}")
    print("=" * 55)

    train(args)