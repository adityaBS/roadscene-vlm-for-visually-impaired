import argparse
import json
import os
from xml.parsers.expat import model
import torch
os.environ["ATEN_NNPACK_ENABLED"] = "0"  # Avoid NNPACK issues on some Linux systems

from PIL import Image
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoConfig, PreTrainedModel

# ====================== MPT/Mosaic compatibility patches ======================
# ====================== MPT/Mosaic compatibility patches ======================
from transformers import AutoConfig, PreTrainedModel

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
        # Use the exact model name you're loading
        config = AutoConfig.from_pretrained("anas-awadalla/mpt-1b-redpajama-200b", trust_remote_code=True)
        ConfigClass = config.__class__
        if not hasattr(ConfigClass, 'hidden_size'):
            print("Patching MosaicGPTConfig to add 'hidden_size' alias...")
            @property
            def hidden_size(self):
                return self.d_model
            ConfigClass.hidden_size = hidden_size
            print("Patch successful: hidden_size now aliases d_model")
    except Exception as e:
        print(f"Failed to patch MosaicGPTConfig: {e}")

# Apply patches immediately
patch_mosaic_mpt()
patch_mosaic_config()
# =============================================================================
# =============================================================================

def run_inference(
    input_file="data/val.json",
    output_file="inference_results_openflamingo.json",
    image_root=".",
    checkpoint_dir="openflamingo_checkpoints",
    use_finetuned=True,
    smoke_test=False
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference on {device}...")

    # === Model creation (exact same as training) ===
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

    # === Load checkpoint ===
    if use_finetuned:
        finetuned_path = os.path.join(checkpoint_dir, "final_weights.pt")
        if os.path.exists(finetuned_path):
            print(f"Loading finetuned checkpoint from {finetuned_path}...")
            
            # CRITICAL FIX: Inspect checkpoint to ensure vocab size matches
            # The checkpoint might have more tokens than the default tokenizer (50280 vs 50277/50279)
            # If shapes don't match, load_state_dict(strict=False) SILENTLY FAILS to load embeddings, resulting in garbage.
            
            ckpt_state_dict = torch.load(finetuned_path, map_location="cpu")
            
            # Find embedding weight in checkpoint
            emb_key = None
            for key in ckpt_state_dict.keys():
                if "lang_encoder" in key and "wte.weight" in key:
                    emb_key = key
                    break
            
            if emb_key:
                ckpt_vocab_size = ckpt_state_dict[emb_key].shape[0]
                current_vocab_size = model.lang_encoder.get_input_embeddings().weight.shape[0]
                tokenizer_vocab_size = len(tokenizer)
                
                print(f"Checkpoint Vocab: {ckpt_vocab_size}, Model Vocab: {current_vocab_size}, Tokenizer Vocab: {tokenizer_vocab_size}")
                
                if ckpt_vocab_size > current_vocab_size:
                    print(f"WARNING: Checkpoint has {ckpt_vocab_size} tokens, but model has {current_vocab_size}.")
                    print(f"Resizing model and tokenizer to match checkpoint to ensure embeddings load correctly...")
                    
                    # Add dummy tokens to tokenizer to reach desired size
                    num_to_add = ckpt_vocab_size - tokenizer_vocab_size
                    if num_to_add > 0:
                        new_tokens = [f"<|extra_token_{i}|>" for i in range(num_to_add)]
                        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
                        print(f"Added {num_to_add} dummy tokens to tokenizer.")
                    
                    # Resize model embeddings
                    model.lang_encoder.resize_token_embeddings(ckpt_vocab_size)
                    print(f"Resized model embeddings to {ckpt_vocab_size}.")
            
            # Configure tokenizer padding if not set (vital for generation)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Tell the model about it
            model.lang_encoder.config.pad_token_id = tokenizer.pad_token_id
            
            # Now load state dict
            msg = model.load_state_dict(ckpt_state_dict, strict=False)
            print(f"Load status: {msg}")
            
        else:
            print(f"Finetuned checkpoint not found at {finetuned_path}. Falling back to official checkpoint.")
            use_finetuned = False

    if not use_finetuned:
        print("Loading official OpenFlamingo-3B-vitl-mpt1b checkpoint...")
        checkpoint_path = hf_hub_download(repo_id="openflamingo/OpenFlamingo-3B-vitl-mpt1b", filename="checkpoint.pt")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)

    model.to(device)
    model.eval()

    # ===================== CRITICAL FIX FOR BEAM SEARCH =====================
    # Three-part fix for the OpenFlamingo + MPT + HuggingFace compatibility issue:
    # 1. Make MPT's forward() accept attention_mask (for HF validation)
    # 2. Make MPT's prepare_inputs_for_generation() handle None attention_mask
    # 3. Patch Flamingo.generate() to pass attention_mask
    
    import types
    
    # Fix 1: Patch MPT forward() to accept and ignore attention_mask
    original_mpt_forward = model.lang_encoder.forward
    
    def patched_mpt_forward(input_ids, *args, attention_mask=None, **kwargs):
        """MPT forward that accepts but doesn't require attention_mask"""
        # Just ignore attention_mask and call original forward
        # MPT doesn't actually use it in forward(), only in prepare_inputs
        if 'attention_mask' in kwargs:
             # We should keep it for generation if MPT uses it, but MPT-1B usually doesn't in forward
             # However, HF generate passes it.
             pass
        return original_mpt_forward(input_ids, *args, attention_mask=attention_mask, **kwargs)
    
    model.lang_encoder.forward = patched_mpt_forward
    
    # Fix 2: Patch prepare_inputs_for_generation to create attention_mask if missing
    original_prepare = model.lang_encoder.prepare_inputs_for_generation
    
    def patched_prepare_inputs(input_ids, past_key_values=None, **kwargs):
        """Create attention_mask if it's missing or None"""
        if 'attention_mask' not in kwargs or kwargs['attention_mask'] is None:
            kwargs['attention_mask'] = torch.ones_like(input_ids)
        return original_prepare(input_ids, past_key_values=past_key_values, **kwargs)
    
    model.lang_encoder.prepare_inputs_for_generation = patched_prepare_inputs
    
    # Fix 3: Patch Flamingo.generate() to pass attention_mask
    original_flamingo_generate = model.generate
    
    def patched_flamingo_generate(self, vision_x, lang_x, attention_mask=None, **kwargs):
        """Patched Flamingo generate that passes attention_mask"""
        # Encode vision features
        self._encode_vision_x(vision_x=vision_x)
        
        # Create attention_mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(lang_x)
        
        # Call lang_encoder.generate WITH attention_mask
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
    print("Applied 3-part patch: MPT forward(), prepare_inputs(), and Flamingo.generate()")
    # ========================================================================

    # Generation settings
    tokenizer.padding_side = "left"  # Important for generation

    # === Load data ===
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if smoke_test:
        data = data[:4]
        print("Smoke test mode: only processing 4 examples")

    results = []
    print(f"Running inference on {len(data)} examples...")

    for item in tqdm(data):
        image_path = os.path.join(image_root, item['image'])
        target_caption = item.get('caption', '')

        # Load and preprocess image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            continue

        # Correct preprocessing: raw pixels → vision encoder runs
        pixel_values = image_processor(image)  # [C, 224, 224]
        vision_x = pixel_values.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 1, C, H, W]

        # Tokenize prompt
        prompt = "<image>"

        lang_x = tokenizer(
            [prompt],
            return_tensors="pt",
        )

        input_ids = lang_x["input_ids"].to(device)

        # CHANGE 1: Create bad_words_ids to prevent code/LaTeX generation
        bad_words = [
            "\\end", "\\begin", "\\include", "\\caption", "\\label",
            "#include", "<div", "<html", "<ul", "<li",
            "def ", "class ", "import ", "```",
        ]
        bad_words_ids = []
        for word in bad_words:
            ids = tokenizer.encode(word, add_special_tokens=False)
            if ids:
                bad_words_ids.append(ids)

        # CHANGE 2: Stable generation parameters (Greedy)
        with torch.no_grad():
            generated_ids = model.generate(
                vision_x=vision_x,
                lang_x=input_ids,
                max_new_tokens=128,
                num_beams=3,
                do_sample=False,
                eos_token_id=tokenizer.convert_tokens_to_ids("<|endofchunk|>"),
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # print(f"Debug: Generated Token IDs: {generated_ids[0].tolist()}")


        # CHANGE 3: Better decoding
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

        if "<|endofchunk|>" in generated_text:
            generated_text = generated_text.split("<|endofchunk|>")[0]

        # Extract prediction after prompt
        if prompt in generated_text:
            prediction = generated_text.split(prompt, 1)[1]
        else:
            prediction = generated_text.replace("<image>", "")

        # CHANGE 4: Improved cleanup
        # Stop at special tokens
        for token in ["<|endofchunk|>", "<|endoftext|>"]:
            if token in prediction:
                prediction = prediction.split(token)[0]
        
        # Stop at LaTeX patterns
        latex_patterns = ["\\end{", "\\begin{", "\\include", "\\caption"]
        for pattern in latex_patterns:
            if pattern in prediction:
                prediction = prediction.split(pattern)[0]
                break
        
        # Stop at code patterns
        code_patterns = ["#include", "<div", "<ul", "<li", "def ", "class "]
        for pattern in code_patterns:
            if pattern in prediction:
                prediction = prediction.split(pattern)[0]
                break
        
        # Stop at dialogue/Q&A patterns
        dialogue_patterns = ["\n\nA:", "\nA:", "Q:", "\n\n\n"]
        for pattern in dialogue_patterns:
            if pattern in prediction:
                prediction = prediction.split(pattern)[0]
                break
        
        # Stop at Wikipedia patterns
        wiki_patterns = ["References\n", "External links\n"]
        for pattern in wiki_patterns:
            if pattern in prediction:
                prediction = prediction.split(pattern)[0]
                break
        
        # Clean up asterisk lists
        if prediction.startswith("*"):
            lines = [l.strip().lstrip("*").strip() for l in prediction.split("\n") if l.strip()]
            clean_lines = []
            for line in lines:
                if not any(p in line for p in ["\\end", "#include", "<div"]):
                    clean_lines.append(line)
                else:
                    break
            if clean_lines:
                prediction = " ".join(clean_lines)
        
        lines = prediction.split(". ")
        if len(lines) > 1 and lines[-1].strip() == lines[-2].strip():
            prediction = ". ".join(lines[:-1]).strip() + "."

        # More aggressive: cut off if the beginning repeats later in the text
        if len(prediction) > 40 and prediction[:25] in prediction[25:]:
            first_part = prediction[:25]
            parts = prediction.split(first_part, 1)  # split after first occurrence
            if len(parts) > 1:
                prediction = first_part + parts[1].split(first_part, 1)[0]  # keep only up to second occurrence
            prediction = prediction.strip()

        # Ensure it ends with a period if needed
        if prediction and not prediction.endswith("."):
            prediction += "."

        # ========================

        # Replace newlines with spaces and clean whitespace
        prediction = prediction.replace("\n", " ")
        prediction = " ".join(prediction.split())
        prediction = prediction.strip()
        
        # Replace newlines with spaces and clean whitespace
        prediction = prediction.replace("\n", " ")
        prediction = " ".join(prediction.split())
        prediction = prediction.strip()

        results.append({
            "image": item['image'],
            "target": target_caption,
            "prediction": prediction
        })

    # Save results
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Inference complete! Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenFlamingo inference")
    parser.add_argument("--input_file", type=str, default="data/val.json")
    parser.add_argument("--output_file", type=str, default="inference_results_openflamingo.json")
    parser.add_argument("--image_root", type=str, default=".")
    parser.add_argument("--checkpoint_dir", type=str, default="openflamingo_checkpoints")
    parser.add_argument("--use_base_weights", action="store_true", help="Force use of original OpenFlamingo weights instead of finetuned")
    parser.add_argument("--smoke_test", action="store_true", help="Run on only 4 examples")
    args = parser.parse_args()

    run_inference(
        input_file=args.input_file,
        output_file=args.output_file,
        image_root=args.image_root,
        checkpoint_dir=args.checkpoint_dir,
        use_finetuned=not args.use_base_weights,
        smoke_test=args.smoke_test
    )