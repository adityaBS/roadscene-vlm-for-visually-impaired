import json
import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
import argparse

def run_inference(
    base_model_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    adapter_path=None,
    input_file="data/sensation_vlm_objects/data/bvpi_sft_test.jsonl",
    output_file="inference_results_smol_bvip.jsonl",
    smoke_test=False,
    image_root=".",
    no_quant=False
):
    """
    Run inference with SmolVLM2, ensuring correct prompt-response separation
    and image format for the processor.
    """
    print("="*60)
    print("SmolVLM2 Robust Inference")
    print("="*60)
    print(f"Base model: {base_model_id}")
    print(f"Adapter: {adapter_path}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print("="*60)
    
    # Check for CUDA
    use_cuda = torch.cuda.is_available()
    device_map = "auto" if use_cuda else "cpu"
    
    # Quantization Config (Critical for 2B+ models on common GPUs)
    bnb_config = None
    if use_cuda and not no_quant:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("Quantization: 4-bit enabled")
    
    # Load Model
    model_kwargs = {"device_map": device_map}
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif use_cuda:
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float32
        
    print(f"Loading base model...")
    model = AutoModelForImageTextToText.from_pretrained(base_model_id, **model_kwargs)
    
    # Load adapter
    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("[SUCCESS] Adapter loaded successfully.")
    else:
        print("[WARNING] No adapter loaded. Running base model.")
        
    # Load processor
    processor = AutoProcessor.from_pretrained(base_model_id)
    model.eval()
    
    # Load data
    data = []
    if input_file.endswith('.jsonl'):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
    if smoke_test:
        print("Smoke test: Limiting to 2 examples.")
        data = data[:2]
        
    results = []
    print(f"\nProcessing {len(data)} items...")
    
    for i, item in enumerate(tqdm(data, desc="Inference")):
        image_path = item["image"]
        # Use EXACT long prompt from training to ensure model follows format
        prompt_text = (
            "Describe the scene for a blind pedestrian. You MUST always output: "
            "Sidewalk position and Road position (relative to the image). Then output "
            "RoadVsSidewalk (is the road left_of/right_of the sidewalk, or crossing_front if the road is in front). "
            "After that, always mention CLOSE objects first with left/center/right. "
            "If no close objects are visible, explicitly say so. Only mention objects you can see."
        )
        
        full_image_path = os.path.join(image_root, image_path)
        try:
            image = Image.open(full_image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {full_image_path}: {e}")
            continue
            
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        # Apply template
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        # FORCE PREFIX: We add ' Sidewalk:' to the end of the prompt to force the model into the correct format.
        # We add a space to match the training format "Assistant: Sidewalk:"
        text += " Sidewalk:"
        
        # SmolVLM2 processor expects images interleaved; for one image, use [[image]]
        inputs = processor(text=text, images=[[image]], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=256,
                do_sample=False,  # Deterministic for evaluation
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
            
        # ROBUST DECODING: Strip input prompt by token index
        # This is safer than splitting on strings which can be altered by tokenizer
        input_len = inputs['input_ids'].shape[1]
        response_ids = generated_ids[0, input_len:]
        # Prepend 'Sidewalk: ' because we added it to the prompt to force the format
        prediction = "Sidewalk: " + processor.decode(response_ids, skip_special_tokens=True).strip()
        
        results.append({
            "image": image_path,
            "target": item.get("response", item.get("caption", "")),
            "prediction": prediction,
            "prompt": prompt_text
        })
        
        # Verbose first example
        if len(results) == 1:
            print(f"\nExample Prediction:\n{prediction}\n")
            
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        if output_file.endswith('.jsonl'):
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
    print(f"\n[SUCCESS] Saved {len(results)} results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, required=True, help="Root folder for images")
    parser.add_argument("--adapter", default=None, help="Path to LoRA checkpoint")
    parser.add_argument("--input_file", default="data/sensation_vlm_objects/data/bvpi_sft_test.jsonl", help="Input JSONL file")
    parser.add_argument("--output", default="inference_results_smol_bvip.jsonl", help="Output JSONL or JSON file")
    parser.add_argument("--smoke_test", action="store_true", help="Run 2 examples only")
    parser.add_argument("--no_quant", action="store_true", help="Disable 4-bit quantization")
    
    args = parser.parse_args()
    
    run_inference(
        adapter_path=args.adapter,
        input_file=args.input_file,
        output_file=args.output,
        smoke_test=args.smoke_test,
        image_root=args.image_root,
        no_quant=args.no_quant
    )
