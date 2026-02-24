import json
import os
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

import json
import os
import torch
import argparse
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def train_vlm(smoke_test=False, output_dir="checkpoints", device=None, no_quant=False, image_root="."):
    # Model ID
    MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
    
    # Load dataset
    data_files = {
        "train": os.path.join("data", "train.json"),
        "validation": os.path.join("data", "val.json")
    }
    dataset = load_dataset("json", data_files=data_files)
    
    if smoke_test:
        print("Running in SMOKE TEST mode (using small subset of data)...")
        dataset["train"] = dataset["train"].select(range(4))
        dataset["validation"] = dataset["validation"].select(range(2))

    # Processor
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Device configuration
    if device is None:
        use_cuda = torch.cuda.is_available()
    else:
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        use_cuda = (device == "cuda") or (device == "auto" and torch.cuda.is_available())
        
    # When using CUDA/QLoRA, device_map="auto" is the most robust setting for Accelerate
    device_map = "auto" if use_cuda else "cpu"

    # Quantization Config (4-bit for efficiency) - Only if CUDA is available usually
    bnb_config = None
    if use_cuda and not no_quant:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("CUDA detected/requested. Using 4-bit quantization with bfloat16.")
    elif use_cuda and no_quant:
        print(f"CUDA detected. Quantization DISABLED by user. Loading in bfloat16.")
    else:
        print(f"Using {device_map}. Skipping 4-bit quantization (loading in float32).")

    # Load Model
    model_kwargs = {
        "device_map": device_map,
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif use_cuda:
         # No quantization but CUDA -> use bfloat16
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        # On CPU, we usually use float32
        model_kwargs["torch_dtype"] = torch.float32

    # AutoModelForVision2Seq is deprecated, use AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        **model_kwargs
    )
    
    # Lora Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"], # Target attention layers
        task_type="CAUSAL_LM" 
    )
    
    # prepare_model_for_kbit_training usually only needed for quantization
    if bnb_config:
        model = prepare_model_for_kbit_training(model)
        
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Data Formatting Function
    def format_example(example):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text", "text": "Describe this road scene."} 
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": example["caption"]}
                ]
            }
        ]
        return messages

    # Training Arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1 if smoke_test else 3,
        max_steps=1 if smoke_test else -1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=1 if smoke_test else 100,
        save_steps=1 if smoke_test else 100,
        learning_rate=2e-4,
        bf16=use_cuda, # RTX 5060 supports bf16
        fp16=False,
        use_cpu=not use_cuda,
        remove_unused_columns=False,
        # dataset_text_field="text", # logic moved to formatting_func or collator
    )
    
    def collate_fn(examples):
        texts = []
        images = []
        for ex in examples:
            # We assume 'image' is a path
            from PIL import Image
            image_path = os.path.join(image_root, ex['image']) 
            
            try:
                img = Image.open(image_path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                img = Image.new('RGB', (224, 224), color='black')
                images.append(img)
                
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this road scene."}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ex['caption']}
                    ]
                }
            ]
            text = processor.apply_chat_template(conversation, add_generation_prompt=False)
            texts.append(text)
            
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        
        labels = batch["input_ids"].clone()
        batch["labels"] = labels
        
        return batch

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collate_fn,
    )

    trainer.train()
    
    final_output = "smoke_checkpoint" if smoke_test else "final_checkpoint"
    trainer.save_model(final_output)
    processor.save_pretrained(final_output)
    print(f"Model saved to {final_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, default=".")
    parser.add_argument("--dataset", default="data/train.json")
    parser.add_argument("--val_dataset", default="data/val.json")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick smoke test")
    parser.add_argument("--output_dir", default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"], help="Device to use for training")
    parser.add_argument("--no_quant", action="store_true", help="Disable quantization (load in float16/bfloat16)")
    args = parser.parse_args()
    
    train_vlm(smoke_test=args.smoke_test, output_dir=args.output_dir, device=args.device, no_quant=args.no_quant, image_root =args.image_root)