import json
import os
import torch
import argparse
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
from PIL import Image

def train_vlm_from_checkpoint(
    checkpoint_path=None,
    smoke_test=False, 
    output_dir="checkpoints_bvpi_finetuned", 
    device=None, 
    no_quant=False, 
    image_root=".",
    train_file="bvpi_sft_train.jsonl",
    val_file="bvpi_sft_val.jsonl"
):
    """
    Train SmolVLM on BVPI dataset, optionally resuming from a previous checkpoint.
    
    Args:
        checkpoint_path: Path to previous checkpoint (e.g., "final_checkpoint" or "checkpoints/checkpoint-500")
                        If None, starts from base model
        smoke_test: Run with minimal data for testing
        output_dir: Where to save new checkpoints
        device: Device to use (auto/cuda/cpu)
        no_quant: Disable quantization
        image_root: Root directory for images
        train_file: Path to training JSONL
        val_file: Path to validation JSONL
    """
    
    # Model ID - base model
    BASE_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    
    print(f"Loading data from {train_file} and {val_file}...")
    dataset = load_dataset("json", data_files={"train": train_file, "validation": val_file})
    
    if smoke_test:
        print("Running in SMOKE TEST mode (using small subset of data)...")
        dataset["train"] = dataset["train"].select(range(4))
        dataset["validation"] = dataset["validation"].select(range(2))

    # Processor - load from checkpoint if available, otherwise from base model
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading processor from checkpoint: {checkpoint_path}")
        processor = AutoProcessor.from_pretrained(checkpoint_path)
    else:
        print(f"Loading processor from base model: {BASE_MODEL_ID}")
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    # Device configuration
    if device is None:
        use_cuda = torch.cuda.is_available()
    else:
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        use_cuda = (device == "cuda") or (device == "auto" and torch.cuda.is_available())
        
    device_map = "auto" if use_cuda else "cpu"

    # Quantization Config (4-bit for efficiency)
    bnb_config = None
    if use_cuda and not no_quant:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("CUDA detected. Using 4-bit quantization with bfloat16.")
    elif use_cuda and no_quant:
        print(f"CUDA detected. Quantization DISABLED by user. Loading in bfloat16.")
    else:
        print(f"Using {device_map}. Skipping 4-bit quantization.")

    # Load Model
    model_kwargs = {
        "device_map": device_map,
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif use_cuda:
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    # Load base model or checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        # First load the base model
        model = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL_ID,
            **model_kwargs
        )
        
        # Then load the PEFT adapter
        if bnb_config:
            model = prepare_model_for_kbit_training(model)
        
        print(f"Loading PEFT adapter from {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
        
        print("Successfully loaded checkpoint!")
        model.print_trainable_parameters()
        
    else:
        if checkpoint_path:
            print(f"Warning: Checkpoint path '{checkpoint_path}' not found. Starting from base model.")
        else:
            print("No checkpoint specified. Starting from base model.")
            
        # Load base model and create new LoRA adapter
        model = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL_ID,
            **model_kwargs
        )
        
        # Create new LoRA Config
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM" 
        )
        
        if bnb_config:
            model = prepare_model_for_kbit_training(model)
            
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Collator - uses the BVPI prompt and response format
    def collate_fn(examples):
        texts = []
        images = []
        for ex in examples:
            image_path = os.path.join(image_root, ex['image']) 
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                # Create a black placeholder image
                img = Image.new('RGB', (224, 224), color='black')
                images.append([img])
            else:
                try:
                    img = Image.open(image_path).convert("RGB")
                    images.append([img])
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    img = Image.new('RGB', (224, 224), color='black')
                    images.append([img])
                
            # Use specific prompt and response from BVPI JSONL
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": ex['prompt']}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ex['response']}
                    ]
                }
            ]
            text = processor.apply_chat_template(conversation, add_generation_prompt=False)
            texts.append(text)
            
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        
        # LABEL MASKING: We only want to train on the response, not the instructions.
        labels = batch["input_ids"].clone()
        
        for i in range(len(texts)):
            full_text = texts[i]
            # SmolVLM2 uses "Assistant: " to separate the user/image from the response in this template.
            assistant_tag = "Assistant: "
            if assistant_tag in full_text:
                prompt_part = full_text.split(assistant_tag)[0] + assistant_tag
                prompt_ids = processor.tokenizer.encode(prompt_part, add_special_tokens=False)
                prompt_len = len(prompt_ids)
                # Mask prompt tokens with -100 so they are ignored by the loss function.
                labels[i, :prompt_len] = -100
            else:
                print(f"Warning: Assistant tag '{assistant_tag}' not found in text. No masking applied.")
        
        batch["labels"] = labels
        
        return batch

    # Training Arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1 if smoke_test else 3,
        max_steps=1 if smoke_test else -1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=1 if smoke_test else 100,
        save_steps=1 if smoke_test else 200,
        save_total_limit=3,  # Keep only last 3 checkpoints to save space
        learning_rate=2e-4,
        bf16=use_cuda,
        fp16=False,
        use_cpu=not use_cuda,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collate_fn,
    )

    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    trainer.train()
    
    # Save final model
    print(f"\nSaving final model to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Training complete! Model saved to {output_dir}")
    
    return trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolVLM on BVPI dataset, optionally from a checkpoint"
    )
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        default=None,
        help="Path to previous checkpoint (e.g., 'final_checkpoint' or 'checkpoints/checkpoint-500'). If not provided, starts from base model."
    )
    parser.add_argument(
        "--image_root", 
        type=str, 
        required=True, 
        help="Path to the root directory of images"
    )
    parser.add_argument(
        "--smoke_test", 
        action="store_true", 
        help="Run a quick smoke test with minimal data"
    )
    parser.add_argument(
        "--output_dir", 
        default="checkpoints_bvpi_finetuned", 
        help="Output directory for new checkpoints"
    )
    parser.add_argument(
        "--device", 
        default=None, 
        choices=["auto", "cuda", "cpu"], 
        help="Device to use for training"
    )
    parser.add_argument(
        "--no_quant", 
        action="store_true", 
        help="Disable quantization"
    )
    parser.add_argument(
        "--train_file", 
        default="bvpi_sft_train.jsonl", 
        help="Path to training JSONL file"
    )
    parser.add_argument(
        "--val_file", 
        default="bvpi_sft_val.jsonl", 
        help="Path to validation JSONL file"
    )
    
    args = parser.parse_args()
    
    train_vlm_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        smoke_test=args.smoke_test, 
        output_dir=args.output_dir, 
        device=args.device, 
        no_quant=args.no_quant, 
        image_root=args.image_root, 
        train_file=args.train_file, 
        val_file=args.val_file
    )
