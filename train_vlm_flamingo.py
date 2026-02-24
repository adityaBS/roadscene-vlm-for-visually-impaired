import argparse
import json
import os
import torch
# Disable NNPACK to avoid "Could not initialize NNPACK! Reason: Unsupported hardware." on some Linux systems
os.environ["ATEN_NNPACK_ENABLED"] = "0"
from PIL import Image
from open_flamingo import create_model_and_transforms
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_constant_schedule_with_warmup, PreTrainedModel, AutoConfig
from torch.optim import AdamW

# Monkey patch for MPT/MosaicGPT embedding resizing
def patch_mosaic_mpt():
    """
    Patch PreTrainedModel to support get/set_input_embeddings for MosaicGPT/MPT models
    that do not implement it, causing create_model_and_transforms to fail during resizing.
    """
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
    print("Patched PreTrainedModel for MosaicGPT compatibility.")

def patch_mosaic_config():
    """
    Patch MosaicGPTConfig to have 'hidden_size' attribute (alias to d_model).
    OpenFlamingo expects 'hidden_size'.
    """
    try:
        config = AutoConfig.from_pretrained("anas-awadalla/mpt-1b-redpajama-200b", trust_remote_code=True)
        ConfigClass = config.__class__
        if not hasattr(ConfigClass, 'hidden_size'):
            print("Patching MosaicGPTConfig to have 'hidden_size'...")
            @property
            def hidden_size(self):
                return self.d_model
            ConfigClass.hidden_size = hidden_size
    except Exception as e:
        print(f"Failed to patch MosaicGPTConfig: {e}")

patch_mosaic_mpt()
patch_mosaic_config()

class VLMDataset(Dataset):
    def __init__(self, json_file, image_dir_root, image_processor, tokenizer):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # IMPROVEMENT 1: Filter out bad examples
        original_count = len(self.data)
        self.data = [item for item in self.data if self._is_valid_example(item)]
        filtered_count = original_count - len(self.data)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} invalid examples (code/LaTeX contamination)")
        
        self.image_dir_root = image_dir_root
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        
        # Add special tokens for Flamingo
        self.tokenizer.padding_side = "right"
    
    def _is_valid_example(self, item):
        """Filter out examples that might teach bad patterns"""
        caption = item.get('caption', '')
        
        # Check length
        words = caption.split()
        if len(words) < 5 or len(words) > 100:
            return False
        
        # CRITICAL: Check for code/LaTeX patterns (data contamination)
        bad_patterns = ["\end", "\begin", "\include", "#include", "<div", "<html",
                       "def ", "class ", "```", "{}", "[]", "<!DOCTYPE"]
        if any(p in caption for p in bad_patterns):
            print(f"Skipping contaminated caption: {caption[:50]}...")
            return False
        
        return True
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir_root, item['image'])
        caption = item['caption']
        
        # Load Image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        # Preprocess Image (raw tensor for vision_encoder)
        processed = self.image_processor(image)  # [C, 224, 224]
        
        # Add dimensions: batch (handled by dataloader), media=1, frames=1
        vision_x = processed.unsqueeze(0)  # [1, C, H, W]     -> num_media dim
        vision_x = vision_x.unsqueeze(0)   # [1, 1, C, H, W]   -> num_frames dim
        
        # IMPROVEMENT 2: Normalize caption and add explicit stop token
        caption = caption.strip()
        if not caption.endswith('.'):
            caption += '.'
        
        # Normalize formatting
        caption = caption.replace(" the next ", " next ")
        
        # Tokenize full text
        # FIXED: Removed <|endoftext|> — only use <|endofchunk|> as stop signal
        full_text = f"<image>{caption}<|endofchunk|>"
        tokenized_full = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        
        input_ids = tokenized_full["input_ids"].squeeze(0)
        attention_mask = tokenized_full["attention_mask"].squeeze(0)
        
        # Create Labels
        labels = input_ids.clone()
        # Mask padding
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # FIXED: Properly mask prompt tokens - match exact tokenization used in full_text
        prompt_text = "<image>"
        tokenized_prompt = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=True  # Critical: matches how full_text is tokenized
        )
        prompt_len = tokenized_prompt["input_ids"].shape[1]
        
        # Safety clamp
        prompt_len = min(prompt_len, labels.shape[0])
        labels[:prompt_len] = -100
        
        return {
            "vision_x": vision_x,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_length": prompt_len  # Store for loss masking
        }

def validate_model(model, dataloader, device, tokenizer, epoch):
    """
    IMPROVEMENT 5: Add validation with generation quality check
    """
    model.eval()
    total_loss = 0
    code_generation_count = 0
    good_generation_count = 0
    
    # Track gate activations during validation
    gate_activations = []
    
    bad_patterns = ["\end", "\begin", "#include", "<div", "def ", "class ", "```"]
    
    print(f"\n{'='*60}")
    print(f"VALIDATION - Epoch {epoch}")
    print('='*60)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            vision_x = batch["vision_x"].to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
            lang_x = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device).bool()
            labels = batch["labels"].to(device)
            
            # Compute loss
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16 if device == "cuda" else torch.bfloat16):
                outputs = model(
                    vision_x=vision_x,
                    lang_x=lang_x,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                if not torch.isnan(loss):
                    total_loss += loss.item()
            
            # Collect gate activations from first batch to monitor visual signal flow
            if batch_idx == 0:
                for name, param in model.named_parameters():
                    if "gated_cross_attn" in name and "gate" in name:
                        gate_activations.append({
                            'name': name,
                            'mean': param.data.mean().item(),
                            'std': param.data.std().item(),
                            'min': param.data.min().item(),
                            'max': param.data.max().item()
                        })
            
            # Check generation quality on first 3 batches
            if batch_idx < 3:
                prompt_text = "<image>"
                prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
                
                try:
                    if hasattr(model, "uncache_media"):
                        model.uncache_media()
                    
                    with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16 if device == "cuda" else torch.bfloat16):
                        generated = model.generate(
                            vision_x=vision_x[:1].to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32),
                            lang_x=prompt_ids,
                            max_new_tokens=50,
                            num_beams=1,
                            temperature=0.7,
                            do_sample=False,
                        )
                    
                    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                    
                    # Check for code patterns
                    has_code = any(pattern in generated_text for pattern in bad_patterns)
                    if has_code:
                        code_generation_count += 1
                        print(f"[Sample {batch_idx+1}] BAD: {generated_text[:80]}...")
                    else:
                        good_generation_count += 1
                        print(f"[Sample {batch_idx+1}] GOOD: {generated_text[:80]}...")
                        
                except Exception as e:
                    print(f"Generation error: {e}")
                finally:
                    if hasattr(model, "uncache_media"):
                        model.uncache_media()
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
    
    # Print gate statistics
    if gate_activations:
        print(f"\n--- Gate Activation Statistics (Validation) ---")
        avg_gate_mean = sum(g['mean'] for g in gate_activations) / len(gate_activations)
        avg_gate_max = sum(g['max'] for g in gate_activations) / len(gate_activations)
        print(f"Average gate value: {avg_gate_mean:.6f}")
        print(f"Average gate max: {avg_gate_max:.6f}")
        print(f"Visual signal flow: ~{avg_gate_mean*100:.2f}% (should be 10-50% for good learning)")
        if avg_gate_mean < 0.05:
            print("⚠️  WARNING: Gates still very low - visual info barely flowing!")
        elif avg_gate_mean > 0.1:
            print("✓ Gates opening - visual information flowing well")
    
    print(f"\nValidation Loss: {avg_loss:.4f}")
    print(f"Generation Quality: {good_generation_count} good, {code_generation_count} code/gibberish")
    if code_generation_count > good_generation_count:
        print("⚠️  WARNING: Model is generating more gibberish than good captions!")
        print("   Consider: 1) Training longer, 2) Checking data quality, 3) Increasing LR")
    elif good_generation_count > 0:
        print("✓ Model generating reasonable captions")
    print('='*60 + '\n')
    
    model.train()
    return avg_loss, gate_activations

def train_openflamingo(
    dataset_path="data/train.json",
    val_dataset_path="data/val.json",
    image_root=".",
    output_dir="openflamingo_checkpoints",
    epochs=10,
    batch_size=2,
    lr=3e-5,
    gate_lr_multiplier=10.0,  # NEW: Separate learning rate for gates
    device="cuda" if torch.cuda.is_available() else "cpu",
    smoke_test=False
):
    print(f"Initializing OpenFlamingo-3B on {device}...")
    
    # Initialize Model and Tokenizer
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
        cross_attn_every_n_layers=1,
        decoder_layers_attr_name="transformer.blocks"
    )
    
    # === RELIABLE VISION ENCODER DIMENSION CHECK ===
    print("=== Vision Encoder Check (fresh model) ===")
    try:
        # Try different paths for CLIP projection
        if hasattr(model.vision_encoder, 'visual'):
            # OpenCLIP structure
            if hasattr(model.vision_encoder.visual, 'proj'):
                proj = model.vision_encoder.visual.proj
            elif hasattr(model.vision_encoder.visual, 'head'):
                proj = model.vision_encoder.visual.head
        elif hasattr(model.vision_encoder, 'proj'):
            proj = model.vision_encoder.proj
        else:
            # Fallback: inspect model structure
            print("Vision encoder structure:")
            for name, module in model.vision_encoder.named_children():
                print(f"  - {name}: {type(module)}")
            proj = None
        
        if proj is not None:
            if isinstance(proj, torch.nn.Parameter):
                print("Visual proj is a Parameter with shape:", proj.shape)
                vision_dim = proj.shape[1]
            else:
                print("Visual proj weight shape:", proj.weight.shape)
                vision_dim = proj.weight.shape[1]
            print(f"Vision encoder embedding dimension: {vision_dim}")
        else:
            print("Could not find projection layer, using default")
            vision_dim = 768  # Default for ViT-L-14
    except Exception as e:
        print(f"Error inspecting vision encoder: {e}")
        vision_dim = 768
    
    try:
        if hasattr(model.vision_encoder, 'positional_embedding'):
            pos_emb = model.vision_encoder.positional_embedding
        elif hasattr(model.vision_encoder, 'visual') and hasattr(model.vision_encoder.visual, 'positional_embedding'):
            pos_emb = model.vision_encoder.visual.positional_embedding
        else:
            pos_emb = None
        
        if pos_emb is not None:
            print("Positional embedding shape:", pos_emb.shape)
    except Exception as e:
        print(f"Could not inspect positional embedding: {e}")
    
    # Fix: MPT tokenizer doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer Pad Token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
    print(f"Tokenizer EOS Token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
    
    # Check endofchunk and endoftext tokens
    eoc_token = "<|endofchunk|>"
    eoc_id = tokenizer.convert_tokens_to_ids(eoc_token)
    eot_id = tokenizer.eos_token_id
    print(f"Token ID for '{eoc_token}': {eoc_id}")
    print(f"Token ID for '<|endoftext|>': {eot_id}")
    
    # Resize embeddings if needed
    if len(tokenizer) > model.lang_encoder.get_input_embeddings().weight.shape[0]:
        print(f"Resizing embeddings from {model.lang_encoder.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
        model.lang_encoder.resize_token_embeddings(len(tokenizer))
    
    # Alias .layers to .transformer.blocks for MPT compatibility
    if hasattr(model.lang_encoder, 'transformer') and hasattr(model.lang_encoder.transformer, 'blocks'):
        print("Aliasing .layers to .transformer.blocks for MPT model compatibility.")
        model.lang_encoder.layers = model.lang_encoder.transformer.blocks
    
    # Patch MPT lang_encoder to ignore labels
    class MPTWrapper(torch.nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.original_model = original_model
            self.config = original_model.config
        
        def forward(self, *args, **kwargs):
            if 'labels' in kwargs:
                kwargs.pop('labels')
            return self.original_model(*args, **kwargs)
        
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.original_model, name)
    
    model.lang_encoder = MPTWrapper(model.lang_encoder)
    
    # Use bfloat16 if cuda is available
    if device == "cuda":
        model.to(torch.bfloat16)
    model.to(device)
    model.train()
    
    # IMPROVEMENT 8: Get bad words IDs to penalize code generation
    bad_words = ["\end", "\begin", "\include", "#include", "<div", "<html",
                 "def ", "class ", "import ", "```"]
    bad_words_ids = []
    for word in bad_words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids and len(ids) > 0:
            bad_words_ids.append(ids[0])  # Get first token ID
    print(f"Bad word token IDs to penalize: {bad_words_ids}")
    
    # Freeze most layers, train only cross_attn
    for name, param in model.named_parameters():
        if "lang_encoder" in name and "gated_cross_attn" not in name:
            param.requires_grad = False
    
    # ============================================================================
    # FIX: PROPER GATE INITIALIZATION
    # ============================================================================
    print("\n" + "="*60)
    print("FIXING GATE INITIALIZATION")
    print("="*60)

    gate_params = []
    for name, param in model.named_parameters():
        if "gated_cross_attn" in name and "gate" in name and param.requires_grad:
            initial_value = param.data.mean().item()
            print(f"Gate {name}: initial value = {initial_value:.6f}, shape = {param.shape}")

            # Only reinitialize if gate is effectively zero (dead)
            if abs(initial_value) < 1e-6:
                print(f"  -> Gate is dead, reinitializing safely")

                # Safe initialization for scalar or 1D gate tensors
                if param.dim() < 2:
                    # For scalar gates: initialize to small positive value (e.g., 0.1 or normal)
                    # Option A: small constant (common in Flamingo papers)
                    torch.nn.init.constant_(param.data, 0.1)

                    # Option B: small normal (more variance)
                    # torch.nn.init.normal_(param.data, mean=0.0, std=0.1)

                    # Option C: zero but with slight positive bias (safe default)
                    # torch.nn.init.constant_(param.data, 0.0)  # keep zero if preferred
                else:
                    # Only use Xavier if tensor has >=2 dimensions
                    torch.nn.init.xavier_normal_(param.data)

                print(f"  -> New value = {param.data.mean().item():.6f}")
            else:
                print(f"  -> Gate already initialized, keeping original value")

            gate_params.append(param)

    print("="*60 + "\n")
    
    # ============================================================================
    # FIX: SEPARATE LEARNING RATES FOR GATES
    # ============================================================================
    # Separate gate parameters from other cross-attention parameters
    gate_param_list = []
    other_trainable_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if "gate" in name and "gated_cross_attn" in name:
            gate_param_list.append(param)
        else:
            other_trainable_params.append(param)
    
    print(f"Gate parameters: {len(gate_param_list)}")
    print(f"Other trainable parameters: {len(other_trainable_params)}")
    
    # Create parameter groups with different learning rates
    param_groups = [
        {'params': other_trainable_params, 'lr': lr, 'name': 'cross_attn'},
        {'params': gate_param_list, 'lr': lr * gate_lr_multiplier, 'name': 'gates'}  # 10x higher LR for gates
    ]
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in other_trainable_params) + sum(p.numel() for p in gate_param_list)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"Base learning rate: {lr}")
    print(f"Gate learning rate: {lr * gate_lr_multiplier} ({gate_lr_multiplier}x base)")
    
    optimizer = AdamW(param_groups, weight_decay=0.01)
    
    # Dataset
    print(f"Loading training data from {dataset_path}...")
    train_dataset = VLMDataset(dataset_path, image_root, image_processor, tokenizer)
    
    val_dataset = None
    if os.path.exists(val_dataset_path):
        print(f"Loading validation data from {val_dataset_path}...")
        val_dataset = VLMDataset(val_dataset_path, image_root, image_processor, tokenizer)
    
    if smoke_test:
        train_dataset.data = train_dataset.data[:4]
        if val_dataset:
            val_dataset.data = val_dataset.data[:4]
        epochs = 2
        print("Smoke test mode: 4 examples, 2 epochs")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    
    # Scheduler
    total_steps = len(train_dataloader) * epochs
    warmup_steps = max(100, int(0.1 * total_steps))
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    print(f"\n{'='*60}")
    print(f"Starting training for {epochs} epochs...")
    print('='*60 + '\n')
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 3
    
    # Training metrics tracking
    training_log = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'gate_stats': [],
        'gradient_stats': [],
        'learning_metrics': []
    }
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{epochs}")
        print('='*60)
        
        epoch_loss = 0
        num_batches = 0
        epoch_gate_gradients = []
        epoch_other_gradients = []
        
        for step, batch in enumerate(train_dataloader):
            vision_x = batch["vision_x"].to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
            lang_x = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device).bool()
            labels = batch["labels"].to(device)
            
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16 if device == "cuda" else torch.bfloat16):
                outputs = model(
                    vision_x=vision_x,
                    lang_x=lang_x,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                
                # Shift for language modeling
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # IMPROVEMENT 9: Standard cross entropy loss
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # IMPROVEMENT 10: Add penalty for code token probabilities
                code_penalty = 0.0
                if len(bad_words_ids) > 0:
                    probs = torch.softmax(shift_logits, dim=-1)
                    for bad_id in bad_words_ids:
                        if bad_id < probs.shape[-1]:
                            code_penalty += probs[..., bad_id].mean()
                    code_penalty = code_penalty / len(bad_words_ids)
                
                # Combine losses
                total_loss = loss + 0.1 * code_penalty  # Small penalty for code tokens
            
            if torch.isnan(total_loss):
                print(f"WARNING: NaN loss at Epoch {epoch+1}, Step {step}. Skipping.")
                optimizer.zero_grad()
                continue
            
            total_loss.backward()
            
            # IMPROVEMENT 11: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Collect gradient statistics for debugging
            gate_grads = []
            other_grads = []
            for name, param in model.named_parameters():
                if param.grad is not None and param.requires_grad:
                    grad_norm = param.grad.norm().item()
                    if "gate" in name and "gated_cross_attn" in name:
                        gate_grads.append(grad_norm)
                    else:
                        other_grads.append(grad_norm)
            
            if gate_grads:
                epoch_gate_gradients.extend(gate_grads)
            if other_grads:
                epoch_other_gradients.extend(other_grads)
            
            if step % 10 == 0:
                penalty_str = f", Code Penalty: {code_penalty.item():.4f}" if code_penalty > 0 else ""
                
                # Calculate perplexity for interpretability
                perplexity = torch.exp(loss).item() if loss.item() < 10 else float('inf')
                
                print(f"\nStep {step}/{len(train_dataloader)}")
                print(f"  Loss: {loss.item():.4f}, Perplexity: {perplexity:.2f}{penalty_str}")
                
                # Gate monitoring
                gate_values = []
                print(f"  Gate Values & Gradients:")
                for name, param in model.named_parameters():
                    if "gated_cross_attn" in name and "gate" in name and param.requires_grad:
                        grad_val = param.grad.norm().item() if param.grad is not None else 0.0
                        gate_mean = param.data.mean().item()
                        gate_values.append(gate_mean)
                        
                        # Visual indicator
                        if gate_mean < 0.05:
                            indicator = "🔴"  # Low - bad
                        elif gate_mean < 0.15:
                            indicator = "🟡"  # Medium - okay
                        else:
                            indicator = "🟢"  # High - good
                        
                        print(f"    {indicator} {name.split('.')[-1]}: value={gate_mean:.6f}, grad={grad_val:.6f}")
                
                # Summary metrics
                if gate_values:
                    avg_gate = sum(gate_values) / len(gate_values)
                    print(f"  Avg Gate Value: {avg_gate:.6f} (~{avg_gate*100:.1f}% visual signal)")
                
                if gate_grads:
                    avg_gate_grad = sum(gate_grads[-len(gate_values):]) / len(gate_values) if gate_values else 0
                    avg_other_grad = sum(other_grads[-10:]) / min(10, len(other_grads)) if other_grads else 0
                    print(f"  Avg Gate Gradient: {avg_gate_grad:.6f}")
                    print(f"  Avg Other Gradient: {avg_other_grad:.6f}")
                    
                    # Check for learning issues
                    if avg_gate_grad < 1e-5:
                        print(f"  ⚠️  WARNING: Gate gradients very small - may not be learning!")
                    elif avg_gate_grad > 0.01:
                        print(f"  ✓ Gate gradients healthy - learning in progress")
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_train_perplexity = torch.exp(torch.tensor(avg_train_loss)).item()
        
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1} SUMMARY")
        print('='*60)
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Training Perplexity: {avg_train_perplexity:.2f}")
        
        # Gate statistics for the epoch
        print(f"\n--- End of Epoch Gate Statistics ---")
        gate_means = []
        gate_stds = []
        for name, param in model.named_parameters():
            if "gated_cross_attn" in name and "gate" in name:
                gate_mean = param.data.mean().item()
                gate_std = param.data.std().item()
                gate_min = param.data.min().item()
                gate_max = param.data.max().item()
                gate_means.append(gate_mean)
                gate_stds.append(gate_std)
                print(f"  {name.split('.')[-1]}: mean={gate_mean:.6f}, std={gate_std:.6f}, range=[{gate_min:.6f}, {gate_max:.6f}]")
        
        if gate_means:
            overall_gate_mean = sum(gate_means) / len(gate_means)
            overall_gate_std = sum(gate_stds) / len(gate_stds)
            print(f"\nOverall Gate Statistics:")
            print(f"  Mean: {overall_gate_mean:.6f} ({overall_gate_mean*100:.2f}% visual signal)")
            print(f"  Std: {overall_gate_std:.6f}")
            
            # Diagnostic feedback
            if overall_gate_mean < 0.05:
                print(f"  🔴 Status: Gates stuck LOW - visual info not flowing well")
                print(f"     → Consider: Increase gate_lr_multiplier or check data quality")
            elif overall_gate_mean < 0.15:
                print(f"  🟡 Status: Gates opening slowly - learning in progress")
            else:
                print(f"  🟢 Status: Gates open - good visual-language integration")
        
        # Gradient statistics
        if epoch_gate_gradients and epoch_other_gradients:
            avg_gate_grad = sum(epoch_gate_gradients) / len(epoch_gate_gradients)
            avg_other_grad = sum(epoch_other_gradients) / len(epoch_other_gradients)
            print(f"\nGradient Statistics (Epoch Average):")
            print(f"  Gate gradients: {avg_gate_grad:.6f}")
            print(f"  Other gradients: {avg_other_grad:.6f}")
            print(f"  Gate/Other ratio: {avg_gate_grad/avg_other_grad:.2f}x" if avg_other_grad > 0 else "  Gate/Other ratio: N/A")
        
        print('='*60)
        
        # Log metrics
        training_log['epochs'].append(epoch + 1)
        training_log['train_loss'].append(avg_train_loss)
        if gate_means:
            training_log['gate_stats'].append({
                'mean': overall_gate_mean,
                'std': overall_gate_std,
                'individual': gate_means
            })
        if epoch_gate_gradients:
            training_log['gradient_stats'].append({
                'gate_avg': sum(epoch_gate_gradients) / len(epoch_gate_gradients),
                'other_avg': sum(epoch_other_gradients) / len(epoch_other_gradients) if epoch_other_gradients else 0
            })
        
        # IMPROVEMENT 12: Validation with generation check
        if val_dataloader:
            val_loss, gate_activations = validate_model(model, val_dataloader, device, tokenizer, epoch+1)
            training_log['val_loss'].append(val_loss)
            
            # Learning progress metrics
            if epoch > 0:
                loss_improvement = training_log['train_loss'][-2] - avg_train_loss
                gate_movement = overall_gate_mean - training_log['gate_stats'][-2]['mean'] if len(training_log['gate_stats']) > 1 else 0
                
                print(f"\n--- Learning Progress Metrics ---")
                print(f"Loss improvement: {loss_improvement:.4f} ({'↓' if loss_improvement > 0 else '↑'})")
                print(f"Gate movement: {gate_movement:.6f} ({'↑' if gate_movement > 0 else '↓'})")
                
                training_log['learning_metrics'].append({
                    'loss_improvement': loss_improvement,
                    'gate_movement': gate_movement
                })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"\n✓ New best validation loss! Saving checkpoint...")
                
                # Save best model
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                if hasattr(model.lang_encoder, "original_model"):
                    model.lang_encoder = model.lang_encoder.original_model
                
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
                
                # Save training log
                import json
                with open(os.path.join(output_dir, "training_log.json"), 'w') as f:
                    json.dump(training_log, f, indent=2)
                
                # Re-wrap for continued training
                model.lang_encoder = MPTWrapper(model.lang_encoder)
            else:
                patience_counter += 1
                print(f"\nNo improvement for {patience_counter} epoch(s)")
                
                if patience_counter >= patience_limit:
                    print(f"⚠️  Early stopping triggered after {patience_limit} epochs without improvement")
                    break
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    # Final training summary
    print("\n--- FINAL TRAINING SUMMARY ---")
    if len(training_log['train_loss']) > 0:
        print(f"Initial Train Loss: {training_log['train_loss'][0]:.4f}")
        print(f"Final Train Loss: {training_log['train_loss'][-1]:.4f}")
        print(f"Total Loss Improvement: {training_log['train_loss'][0] - training_log['train_loss'][-1]:.4f}")
    
    if len(training_log['gate_stats']) > 1:
        initial_gate = training_log['gate_stats'][0]['mean']
        final_gate = training_log['gate_stats'][-1]['mean']
        print(f"\nGate Evolution:")
        print(f"  Initial: {initial_gate:.6f} ({initial_gate*100:.2f}% visual signal)")
        print(f"  Final: {final_gate:.6f} ({final_gate*100:.2f}% visual signal)")
        print(f"  Total Movement: {final_gate - initial_gate:.6f}")
        
        if final_gate < 0.05:
            print(f"  🔴 Status: Gates never opened - model NOT learning from images")
            print(f"     DEBUG: Check data quality, increase gate_lr_multiplier to 20+")
        elif final_gate < 0.15:
            print(f"  🟡 Status: Gates partially open - some visual learning")
            print(f"     TIP: Consider training longer or slightly higher LR")
        else:
            print(f"  🟢 Status: Gates fully open - strong visual-language learning!")
    
    if len(training_log['val_loss']) > 0:
        print(f"\nBest Validation Loss: {min(training_log['val_loss']):.4f}")
        print(f"Final Validation Loss: {training_log['val_loss'][-1]:.4f}")
    
    print("\n" + "="*60)
    
    print("\n=== Final Vision Encoder Check Before Saving ===")
    try:
        if hasattr(model.vision_encoder, 'positional_embedding'):
            pos_emb = model.vision_encoder.positional_embedding
        elif hasattr(model.vision_encoder, 'visual') and hasattr(model.vision_encoder.visual, 'positional_embedding'):
            pos_emb = model.vision_encoder.visual.positional_embedding
        else:
            pos_emb = None
        
        if pos_emb is not None:
            print("Positional embedding shape:", pos_emb.shape)
        
        if hasattr(model.vision_encoder, 'visual'):
            if hasattr(model.vision_encoder.visual, 'proj'):
                proj = model.vision_encoder.visual.proj
            elif hasattr(model.vision_encoder.visual, 'head'):
                proj = model.vision_encoder.visual.head
        elif hasattr(model.vision_encoder, 'proj'):
            proj = model.vision_encoder.proj
        else:
            proj = None
        
        if proj is not None:
            if isinstance(proj, torch.nn.Parameter):
                proj_shape = proj.shape
            else:
                proj_shape = proj.weight.shape
            print("Proj shape:", proj_shape)
    except Exception as e:
        print(f"Could not inspect vision encoder: {e}")
    print("===============================================\n")
    
    # Save final model
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Unwrap MPTWrapper before saving
    if hasattr(model.lang_encoder, "original_model"):
        print("Unwrapping MPTWrapper before saving...")
        model.lang_encoder = model.lang_encoder.original_model
    
    print("Saving final model weights...")
    torch.save(model.state_dict(), os.path.join(output_dir, "final_weights.pt"))
    print(f"Saved final weights to {output_dir}/final_weights.pt")
    
    # Save final training log
    import json
    with open(os.path.join(output_dir, "training_log.json"), 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"Saved training log to {output_dir}/training_log.json")
    
    if best_val_loss < float('inf'):
        print(f"Best model saved at {output_dir}/best_model.pt (Val Loss: {best_val_loss:.4f})")
    
    print("\n--- DEBUGGING PARAMETERS ---")
    print("For debugging training issues, check these in training_log.json:")
    print("  • gate_stats: Track how gates evolved (mean, std, individual values)")
    print("  • gradient_stats: Monitor gradient flow (gate_avg vs other_avg)")
    print("  • learning_metrics: Loss improvement and gate movement per epoch")
    print("  • train_loss/val_loss: Overall convergence trajectory")
    print("\nKey diagnostics:")
    print("  • If gates stay < 0.05: Increase gate_lr_multiplier (try 20-50)")
    print("  • If gradients < 1e-5: Check data quality or increase base LR")
    print("  • If val_loss >> train_loss: Reduce overfitting (more data/regularization)")
    print("  • If loss plateaus early: May need more training data or longer training")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, default=".")
    parser.add_argument("--dataset", default="data/train.json")
    parser.add_argument("--val_dataset", default="data/val.json")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--output_dir", default="openflamingo_checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--gate_lr_multiplier", type=float, default=10.0, help="Learning rate multiplier for attention gates")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    
    args = parser.parse_args()
    
    train_openflamingo(
        image_root=args.image_root,
        dataset_path=args.dataset,
        val_dataset_path=args.val_dataset,
        smoke_test=args.smoke_test,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        gate_lr_multiplier=args.gate_lr_multiplier,
        batch_size=args.batch_size
    )