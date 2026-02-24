import argparse
import json
import os
import re
import types
import torch
os.environ["ATEN_NNPACK_ENABLED"] = "0"  # Avoid NNPACK issues on some Linux systems

from PIL import Image
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoConfig, PreTrainedModel

# ====================== MPT/Mosaic compatibility patches ======================
def patch_mosaic_mpt():
    """Patch to support get/set_input_embeddings for Mosaic MPT models"""
    orig_get = PreTrainedModel.get_input_embeddings
    orig_set = PreTrainedModel.set_input_embeddings

    def get_input_embeddings(self):
        if hasattr(self, 'transformer') and hasattr(self.transformer, 'wte'):
            return self.transformer.wte
        return orig_get(self)

    def set_input_embeddings(self, value):
        if hasattr(self, 'transformer') and hasattr(self.transformer, 'wte'):
            self.transformer.wte = value
            return
        return orig_set(self, value)

    PreTrainedModel.get_input_embeddings = get_input_embeddings
    PreTrainedModel.set_input_embeddings = set_input_embeddings
    print("Patched PreTrainedModel for MosaicGPT input embeddings.")


def patch_mosaic_config():
    """Patch MosaicGPTConfig to expose 'hidden_size' alias for d_model"""
    try:
        config = AutoConfig.from_pretrained("anas-awadalla/mpt-1b-redpajama-200b", trust_remote_code=True)
        ConfigClass = config.__class__
        if not hasattr(ConfigClass, 'hidden_size'):
            print("Patching MosaicGPTConfig to add 'hidden_size' alias...")
            @property
            def hidden_size(self):
                return self.d_model
            ConfigClass.hidden_size = hidden_size
    except Exception as e:
        print(f"Failed to patch MosaicGPTConfig: {e}")


patch_mosaic_mpt()
patch_mosaic_config()
# =============================================================================


def load_data(data_file):
    """
    Load data from JSON or JSONL file.

    Supported formats:
      - JSONL: one JSON object per line
          {"image": "...", "prompt": "...", "response": "..."}
      - JSON array:
          [{"image": "...", "caption": "..."}, ...]
    
    Returns (data: list[dict], is_prompted: bool)
    """
    data = []
    ext = os.path.splitext(data_file)[1].lower()

    with open(data_file, 'r', encoding='utf-8') as f:
        if ext == '.jsonl':
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        else:
            content = f.read().strip()
            try:
                parsed = json.loads(content)
                data = parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

    if not data:
        raise ValueError(f"No data loaded from {data_file}")

    sample = data[0]
    is_prompted = 'prompt' in sample and 'response' in sample
    mode = "prompt-response" if is_prompted else "caption-only"
    print(f"Loaded {len(data)} examples from {data_file}  [mode: {mode}]")
    return data, is_prompted


def build_prompt(item, is_prompted):
    """
    Build the generation prompt string for a single data item.

    Prompted mode : "<image>User: {prompt}\\nAssistant: "
    Caption mode  : "<image>"
    """
    if is_prompted:
        return f"<image>User: {item['prompt'].strip()}\nAssistant: "
    return "<image>"


def get_reference(item, is_prompted):
    """Return the ground-truth text for evaluation display."""
    if is_prompted:
        return item.get('response', '')
    return item.get('caption', '')


def clean_prediction(raw_text, prompt_text, is_prompted):
    """
    Strip the prompt prefix and remove contamination patterns from raw decoded output.
    Returns the raw response text — structured parsing is done separately.
    """
    prediction = raw_text

    # 1. Remove prompt prefix
    if prompt_text in prediction:
        prediction = prediction.split(prompt_text, 1)[1]
    else:
        prediction = prediction.replace("<image>", "")
        if is_prompted and "\nAssistant: " in prediction:
            prediction = prediction.split("\nAssistant: ", 1)[-1]

    # 2. Cut at special tokens
    for token in ["<|endofchunk|>", "<|endoftext|>"]:
        if token in prediction:
            prediction = prediction.split(token)[0]

    # 3. Cut at contamination patterns
    stop_patterns = [
        "\\end{", "\\begin{", "\\include", "\\caption",
        "#include", "<div", "<ul", "<li", "def ", "class ",
        "\n\nA:", "\nA:", "Q:", "\n\n\n",
        "References\n", "External links\n",
    ]
    for pattern in stop_patterns:
        if pattern in prediction:
            prediction = prediction.split(pattern)[0]
            break

    # 4. Clean asterisk lists
    if prediction.strip().startswith("*"):
        lines = [l.strip().lstrip("*").strip() for l in prediction.split("\n") if l.strip()]
        clean_lines = [l for l in lines if not any(p in l for p in ["\\end", "#include", "<div"])]
        prediction = " ".join(clean_lines) if clean_lines else prediction

    # 5. Remove duplicate trailing sentence
    sentences = prediction.split(". ")
    if len(sentences) > 1 and sentences[-1].strip() == sentences[-2].strip():
        prediction = ". ".join(sentences[:-1]).strip() + "."

    # 6. Cut runaway repetition
    if len(prediction) > 40 and prediction[:25] in prediction[25:]:
        first_part = prediction[:25]
        parts = prediction.split(first_part, 1)
        if len(parts) > 1:
            prediction = first_part + parts[1].split(first_part, 1)[0]

    # 7. Normalise whitespace
    prediction = prediction.replace("\n", " ")
    prediction = " ".join(prediction.split()).strip()

    # 8. Ensure terminal punctuation
    if prediction and prediction[-1] not in ".!?":
        prediction += "."

    return prediction


# ---------------------------------------------------------------------------
# Structured output: parsing and enforcement
# ---------------------------------------------------------------------------

# Valid values for each structured field
VALID_SIDEWALK_POS   = {"left", "center", "right", "none"}
VALID_ROAD_POS       = {"left", "center", "right", "front", "none"}
VALID_ROAD_VS_SW     = {"left_of", "right_of", "crossing_front", "unknown"}

# Patterns to extract each field from free-form or structured text
_SIDEWALK_RE     = re.compile(r'sidewalk\s*:\s*([a-z_]+)', re.IGNORECASE)
_ROAD_POS_RE     = re.compile(r'road\s*:\s*([a-z_]+)', re.IGNORECASE)
_ROAD_VS_SW_RE   = re.compile(
    r'road\s*vs\s*sidewalk\s*:\s*([a-z_]+)',
    re.IGNORECASE
)
_NEARBY_RE       = re.compile(r'nearby\s*:\s*(.+?)(?:\.|far:|context:|$)', re.IGNORECASE | re.DOTALL)
_FAR_RE          = re.compile(r'far\s*:\s*(.+?)(?:\.|context:|$)',          re.IGNORECASE | re.DOTALL)
_CONTEXT_RE      = re.compile(r'context\s*:\s*(.+)',                         re.IGNORECASE | re.DOTALL)

# Fallback spatial inference from free-form text.
# Order matters: check unambiguous/specific phrases first so "centered" wins
# over a later mention of "on the left" when parsing sidewalk position.
_POSITION_WORDS_ORDERED = [
    ("center", ["centered", "centred", "center", "middle", "straight ahead"]),
    ("front",  ["in front", "crossing_front", "ahead", "forward"]),
    ("left",   ["on the left", "left side", " left "]),
    ("right",  ["on the right", "right side", " right "]),
]


def _find_position(text, valid_set):
    """
    Return the first valid position found in text using ordered phrase matching.
    More specific / unambiguous phrases (centered, in front) are checked before
    generic directional words (left, right) to avoid false matches.
    """
    text_lower = " " + text.lower() + " "   # pad so word-boundary spaces work
    for pos, phrases in _POSITION_WORDS_ORDERED:
        if pos not in valid_set:
            continue
        if any(p in text_lower for p in phrases):
            return pos
    return None


def parse_structured_fields(text):
    """
    Parse the five expected structured fields from model output text.
    Works on both well-formatted responses:
        "Sidewalk: center. Road: left. RoadVsSidewalk: left_of. ..."
    and free-form responses:
        "The sidewalk is centered... a road on the left..."

    Returns a dict with keys:
        sidewalk_pos, road_pos, road_vs_sidewalk,
        nearby, far, context,
        format_ok (bool — True if all three mandatory fields were present)
    """
    fields = {
        "sidewalk_pos":     None,
        "road_pos":         None,
        "road_vs_sidewalk": None,
        "nearby":           None,
        "far":              None,
        "context":          None,
        "format_ok":        False,
    }

    # --- Mandatory structured fields via regex ---
    m = _SIDEWALK_RE.search(text)
    if m:
        fields["sidewalk_pos"] = m.group(1).strip().lower()

    m = _ROAD_POS_RE.search(text)
    if m:
        fields["road_pos"] = m.group(1).strip().lower()

    m = _ROAD_VS_SW_RE.search(text)
    if m:
        fields["road_vs_sidewalk"] = m.group(1).strip().lower()

    # --- Optional detail fields ---
    m = _NEARBY_RE.search(text)
    if m:
        fields["nearby"] = m.group(1).strip().rstrip(".")

    m = _FAR_RE.search(text)
    if m:
        fields["far"] = m.group(1).strip().rstrip(".")

    m = _CONTEXT_RE.search(text)
    if m:
        fields["context"] = m.group(1).strip().rstrip(".")

    # --- Fallback: infer from free-form text ---
    if fields["sidewalk_pos"] is None:
        fields["sidewalk_pos"] = _find_position(text, VALID_SIDEWALK_POS) or "unknown"

    if fields["road_pos"] is None:
        # For road position, search the text for explicit "road … {direction}" patterns
        # first, then fall back to general position search.
        road_context_re = re.compile(
            r'(?:road|street|lane|traffic|cars?|vehicles?)\s+(?:\w+\s+){0,4}?'
            r'(?:on\s+the\s+)?(left|right|front|center|ahead)',
            re.IGNORECASE
        )
        rm = road_context_re.search(text)
        if rm:
            val = rm.group(1).strip().lower()
            val = "front" if val == "ahead" else val
            fields["road_pos"] = val if val in VALID_ROAD_POS else "unknown"
        else:
            # Last resort: search the SECOND HALF of the text to avoid matching
            # sidewalk-describing words that appear first
            mid   = len(text) // 2
            later = text[mid:]
            fields["road_pos"] = _find_position(later, VALID_ROAD_POS) or "unknown"

    if fields["road_vs_sidewalk"] is None:
        rp = fields["road_pos"]
        fields["road_vs_sidewalk"] = {
            "front":  "crossing_front",
            "left":   "left_of",
            "right":  "right_of",
        }.get(rp, "unknown")

    # format_ok = all three mandatory fields found directly by structured regex
    fields["format_ok"] = (
        _SIDEWALK_RE.search(text)   is not None and
        _ROAD_POS_RE.search(text)   is not None and
        _ROAD_VS_SW_RE.search(text) is not None
    )

    return fields


def enforce_structured_format(prediction_text, parsed):
    """
    If the model output is missing the mandatory structured header, prepend it.
    This ensures downstream consumers always get the expected format regardless
    of whether the model followed instructions.

    Output format (mirrors training data):
        Sidewalk: {pos}. Road: {pos}. RoadVsSidewalk: {rel}.
        Nearby: {nearby}. Far: {far}. Context: {context}
    """
    if parsed["format_ok"]:
        # Already well-structured — return as-is
        return prediction_text

    # Build mandatory header from parsed/inferred fields
    header = (
        f"Sidewalk: {parsed['sidewalk_pos']}. "
        f"Road: {parsed['road_pos']}. "
        f"RoadVsSidewalk: {parsed['road_vs_sidewalk']}."
    )

    # Add nearby if available, else default
    nearby = parsed.get("nearby") or "no important close objects visible"
    far     = parsed.get("far")
    context = parsed.get("context") or prediction_text.strip()

    body = f" Nearby: {nearby}."
    if far:
        body += f" Far: {far}."
    body += f" Context: {context}"

    return (header + body).strip()


# ---------------------------------------------------------------------------
# Field-level accuracy metrics
# ---------------------------------------------------------------------------

def compute_structured_metrics(results):
    """
    Compare predicted structured fields against ground-truth metadata fields
    (sidewalk_pos, road_pos, road_vs_sidewalk) when present in the dataset.

    Returns a dict of per-field accuracy and an overall score.
    """
    counts = {"sidewalk_pos": 0, "road_pos": 0, "road_vs_sidewalk": 0, "total": 0}
    correct = {"sidewalk_pos": 0, "road_pos": 0, "road_vs_sidewalk": 0}
    format_ok_count = 0

    for r in results:
        if "sidewalk_pos" not in r:   # no ground-truth metadata → skip
            continue
        counts["total"] += 1
        if r.get("format_ok"):
            format_ok_count += 1

        pred_fields = r.get("parsed_fields", {})
        for field in ("sidewalk_pos", "road_pos", "road_vs_sidewalk"):
            gt  = r.get(field, "").strip().lower()
            pred = pred_fields.get(field, "").strip().lower()
            counts[field] += 1
            if gt and pred and gt == pred:
                correct[field] += 1

    if counts["total"] == 0:
        return {}

    metrics = {
        "total_examples":       counts["total"],
        "format_compliance_%":  round(100 * format_ok_count / counts["total"], 1),
    }
    for field in ("sidewalk_pos", "road_pos", "road_vs_sidewalk"):
        if counts[field] > 0:
            metrics[f"{field}_accuracy_%"] = round(100 * correct[field] / counts[field], 1)

    scored_fields = [f for f in ("sidewalk_pos", "road_pos", "road_vs_sidewalk") if counts[f] > 0]
    if scored_fields:
        metrics["overall_field_accuracy_%"] = round(
            sum(metrics[f"{f}_accuracy_%"] for f in scored_fields) / len(scored_fields), 1
        )

    return metrics


def apply_generation_patches(model):
    """
    Apply the three-part patch needed for OpenFlamingo + MPT + HuggingFace
    beam-search / greedy generation compatibility.
    """
    # Patch 1: MPT forward() accepts attention_mask keyword
    original_mpt_forward = model.lang_encoder.forward

    def patched_mpt_forward(input_ids, *args, attention_mask=None, **kwargs):
        return original_mpt_forward(input_ids, *args, attention_mask=attention_mask, **kwargs)

    model.lang_encoder.forward = patched_mpt_forward

    # Patch 2: prepare_inputs_for_generation creates attention_mask when absent
    original_prepare = model.lang_encoder.prepare_inputs_for_generation

    def patched_prepare_inputs(input_ids, past_key_values=None, **kwargs):
        if 'attention_mask' not in kwargs or kwargs['attention_mask'] is None:
            kwargs['attention_mask'] = torch.ones_like(input_ids)
        return original_prepare(input_ids, past_key_values=past_key_values, **kwargs)

    model.lang_encoder.prepare_inputs_for_generation = patched_prepare_inputs

    # Patch 3: Flamingo.generate() passes attention_mask to lang_encoder.generate()
    def patched_flamingo_generate(self, vision_x, lang_x, attention_mask=None, **kwargs):
        self._encode_vision_x(vision_x=vision_x)

        if attention_mask is None:
            attention_mask = torch.ones_like(lang_x)

        output = self.lang_encoder.generate(
            inputs=lang_x,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Clear conditioned layers
        for method_name in ['_clear_conditioned_layers', 'clear_conditioned_layers',
                            '_reset_conditioned_layers', 'reset_conditioned_layers']:
            if hasattr(self, method_name):
                getattr(self, method_name)()
                break
        else:
            if hasattr(self, 'lang_encoder') and hasattr(self.lang_encoder, 'gated_cross_attn_layers'):
                for layer in self.lang_encoder.gated_cross_attn_layers:
                    if hasattr(layer, '_media_locations'):
                        layer._media_locations = None

        return output

    model.generate = types.MethodType(patched_flamingo_generate, model)
    print("Applied 3-part generation patch (MPT forward, prepare_inputs, Flamingo.generate).")


def run_inference(
    input_file="data/val.jsonl",
    output_file="inference_results_openflamingo.json",
    image_root=".",
    checkpoint_dir="openflamingo_checkpoints",
    use_finetuned=True,
    smoke_test=False,
    max_new_tokens=128,
    num_beams=3,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference on {device}...")

    # ---- Load data first so we know the mode before building prompts ----
    data, is_prompted = load_data(input_file)

    if smoke_test:
        data = data[:4]
        print("Smoke test: processing 4 examples only")

    # ---- Build model ----
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
        cross_attn_every_n_layers=1,
        decoder_layers_attr_name="transformer.blocks"
    )

    # MPT compatibility alias
    if hasattr(model.lang_encoder, 'transformer') and hasattr(model.lang_encoder.transformer, 'blocks'):
        model.lang_encoder.layers = model.lang_encoder.transformer.blocks

    # ---- Load checkpoint ----
    if use_finetuned:
        finetuned_path = os.path.join(checkpoint_dir, "final_weights.pt")
        best_path      = os.path.join(checkpoint_dir, "best_model.pt")

        # Prefer best_model.pt if it exists
        ckpt_path = best_path if os.path.exists(best_path) else finetuned_path

        if os.path.exists(ckpt_path):
            print(f"Loading finetuned checkpoint: {ckpt_path}")
            ckpt_state_dict = torch.load(ckpt_path, map_location="cpu")

            # Vocab-size alignment (checkpoint may have more tokens than default model)
            emb_key = next(
                (k for k in ckpt_state_dict if "lang_encoder" in k and "wte.weight" in k),
                None
            )
            if emb_key:
                ckpt_vocab  = ckpt_state_dict[emb_key].shape[0]
                model_vocab = model.lang_encoder.get_input_embeddings().weight.shape[0]
                tok_vocab   = len(tokenizer)
                print(f"Vocab sizes — checkpoint: {ckpt_vocab}  model: {model_vocab}  tokenizer: {tok_vocab}")

                if ckpt_vocab > model_vocab:
                    num_to_add = ckpt_vocab - tok_vocab
                    if num_to_add > 0:
                        new_tokens = [f"<|extra_token_{i}|>" for i in range(num_to_add)]
                        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
                        print(f"Added {num_to_add} dummy tokens to tokenizer.")
                    model.lang_encoder.resize_token_embeddings(ckpt_vocab)
                    print(f"Resized model embeddings → {ckpt_vocab}")

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.lang_encoder.config.pad_token_id = tokenizer.pad_token_id

            msg = model.load_state_dict(ckpt_state_dict, strict=False)
            print(f"Checkpoint load status: {msg}")
        else:
            print(f"No finetuned checkpoint found in {checkpoint_dir}. Falling back to base weights.")
            use_finetuned = False

    if not use_finetuned:
        print("Loading official OpenFlamingo-3B-vitl-mpt1b checkpoint…")
        ckpt_path = hf_hub_download(
            repo_id="openflamingo/OpenFlamingo-3B-vitl-mpt1b",
            filename="checkpoint.pt"
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    # ---- Apply generation patches ----
    apply_generation_patches(model)

    # Generation padding direction: left-pad so new tokens grow to the right
    tokenizer.padding_side = "left"

    # EOS / stop token
    eoc_token_id = tokenizer.convert_tokens_to_ids("<|endofchunk|>")

    # ---- Inference loop ----
    results = []
    format_warnings = 0
    print(f"\nRunning inference on {len(data)} examples…")

    for item in tqdm(data):
        image_path = os.path.join(image_root, item['image'])

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load {image_path}: {e}")
            results.append({
                "image":      item['image'],
                "prompt":     item.get('prompt', ''),
                "reference":  get_reference(item, is_prompted),
                "prediction": "[image load error]",
                "format_ok":  False,
            })
            continue

        # Preprocess image → [1, 1, 1, C, H, W]
        vision_x = image_processor(image).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)

        # Build prompt text
        prompt_text = build_prompt(item, is_prompted)

        lang_x         = tokenizer([prompt_text], return_tensors="pt")
        input_ids      = lang_x["input_ids"].to(device)
        attention_mask = lang_x["attention_mask"].to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                vision_x=vision_x,
                lang_x=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                eos_token_id=eoc_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode (keep special tokens for splitting)
        raw_text   = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        prediction = clean_prediction(raw_text, prompt_text, is_prompted)

        # ---- Structured output parsing ----
        parsed = parse_structured_fields(prediction) if is_prompted else {}

        # Enforce structured format — prepend header if model skipped it
        if is_prompted:
            prediction = enforce_structured_format(prediction, parsed)
            if not parsed.get("format_ok"):
                format_warnings += 1

        result = {
            "image":      item['image'],
            "reference":  get_reference(item, is_prompted),
            "prediction": prediction,
        }

        if is_prompted:
            result["prompt"]        = item.get('prompt', '')
            result["format_ok"]     = parsed.get("format_ok", False)
            result["parsed_fields"] = {
                k: parsed[k]
                for k in ("sidewalk_pos", "road_pos", "road_vs_sidewalk", "nearby", "far", "context")
            }

        # Carry ground-truth metadata fields for evaluation
        for extra_key in ('sidewalk_pos', 'road_pos', 'road_vs_sidewalk'):
            if extra_key in item:
                result[extra_key] = item[extra_key]

        results.append(result)

    # ---- Save results ----
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nInference complete! {len(results)} results saved → {output_file}")

    # ---- Structured accuracy metrics ----
    if is_prompted:
        if format_warnings:
            print(f"⚠️  {format_warnings}/{len(results)} predictions were missing structured format "
                  f"— headers were auto-enforced.")
        else:
            print(f"✓ All predictions followed the structured format.")

        metrics = compute_structured_metrics(results)
        if metrics:
            print("\n--- Structured Field Accuracy ---")
            print(f"  Total examples evaluated : {metrics['total_examples']}")
            print(f"  Format compliance        : {metrics['format_compliance_%']}%")
            print(f"  sidewalk_pos accuracy    : {metrics.get('sidewalk_pos_accuracy_%', 'N/A')}%")
            print(f"  road_pos accuracy        : {metrics.get('road_pos_accuracy_%', 'N/A')}%")
            print(f"  road_vs_sidewalk accuracy: {metrics.get('road_vs_sidewalk_accuracy_%', 'N/A')}%")
            print(f"  Overall field accuracy   : {metrics.get('overall_field_accuracy_%', 'N/A')}%")

            # Save metrics alongside results
            metrics_file = output_file.replace(".json", "_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\nMetrics saved → {metrics_file}")

        # Sample predictions
        print("\n--- Sample predictions (first 3) ---")
        for r in results[:3]:
            fmt = "✓" if r.get("format_ok") else "⚠"
            print(f"\n  [{fmt}] {r['image']}")
            print(f"  Reference  : {r['reference'][:120]}")
            print(f"  Prediction : {r['prediction'][:120]}")
            if r.get("parsed_fields"):
                pf = r["parsed_fields"]
                print(f"  Parsed     : sidewalk={pf['sidewalk_pos']}  "
                      f"road={pf['road_pos']}  "
                      f"rel={pf['road_vs_sidewalk']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenFlamingo inference (caption-only or prompt-response)")
    parser.add_argument("--input_file",      type=str,   default="data/val.jsonl",
                        help=".jsonl with {image, prompt, response} OR .json with {image, caption}")
    parser.add_argument("--output_file",     type=str,   default="inference_results_openflamingo.json")
    parser.add_argument("--image_root",      type=str,   default=".")
    parser.add_argument("--checkpoint_dir",  type=str,   default="openflamingo_checkpoints")
    parser.add_argument("--use_base_weights",action="store_true",
                        help="Force use of original OpenFlamingo weights instead of finetuned")
    parser.add_argument("--smoke_test",      action="store_true",
                        help="Run on only 4 examples for quick testing")
    parser.add_argument("--max_new_tokens",  type=int,   default=128)
    parser.add_argument("--num_beams",       type=int,   default=3)
    args = parser.parse_args()

    run_inference(
        input_file=args.input_file,
        output_file=args.output_file,
        image_root=args.image_root,
        checkpoint_dir=args.checkpoint_dir,
        use_finetuned=not args.use_base_weights,
        smoke_test=args.smoke_test,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )