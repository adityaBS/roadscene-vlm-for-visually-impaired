import json
import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
import argparse

def run_inference(
    base_model_id="HuggingFaceTB/SmolVLM-Instruct",
    adapter_path="final_checkpoint",
    input_file="data/val.json",
    output_file="inference_results.json",
    smoke_test=False,
    image_root="."
):
    print(f"Loading base model: {base_model_id}...")
    # Load base model
    # Check for CUDA
    use_cuda = torch.cuda.is_available()
    
    # Quantization Config
    bnb_config = None
    if use_cuda:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("CUDA detected. Using 4-bit quantization.")

    # Load base model
    # AutoModelForVision2Seq is deprecated
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        device_map="cuda" if use_cuda else "cpu",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
    )
    
    print(f"Loading adapter from {adapter_path}...")
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Warning: Could not load adapter from {adapter_path}. Running with base model only. Error: {e}")

    processor = AutoProcessor.from_pretrained(base_model_id)
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if smoke_test:
        print("Running in SMOKE TEST mode (limit to 2 examples)...")
        data = data[:2]
        
    results = []
    print(f"Running inference on {len(data)} examples...")
    
    for item in tqdm(data):
        image_path = item["image"]
        target_caption = item.get("caption", "")
        
        # Load image
        full_image_path = os.path.join(image_root, image_path)
        try:
            image = Image.open(full_image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {full_image_path}: {e}")
            continue

        # Create prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this road scene."}
                ]
            }
        ]
        
        # Prepare inputs
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=image, return_tensors="pt")
        inputs = inputs.to(model.device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)
            
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Extract assistant response (simple parsing)
        prediction = generated_texts[0]
        
        # Simple cleanup if the prompt is repeated in output (common in some configs)
        # logic depends on specific model behavior
        
        result_entry = {
            "image_path": image_path,
            "target": target_caption,
            "prediction": prediction
        }
        results.append(result_entry)
        print(f"Processed {len(results)}/{len(data)}")

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Saved inference results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="HuggingFaceTB/SmolVLM-Instruct", help="Base model ID")
    parser.add_argument("--adapter", default="final_checkpoint", help="Path to LoRA adapter")
    parser.add_argument("--input", default="data/val.json", help="Input JSON file")
    parser.add_argument("--output", default="inference_results.json", help="Output JSON file")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick smoke test")
    parser.add_argument("--image_root", type=str, default=".")
    args = parser.parse_args()
    
    
    run_inference(args.base_model, args.adapter, args.input, args.output, args.smoke_test, args.image_root)