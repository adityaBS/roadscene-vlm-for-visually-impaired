import json
import argparse
import sys
import os

def evaluate_vlm(results_file="inference_results.json", image_root="."):
    print(f"Loading results from {results_file}...")
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {results_file} not found.")
        return

    # Prepare data for COCO Eval
    # COCO Eval expects a specific format.
    # We will use pycocoevalcap directly if possible, or build the dicts.
    
    # 1. Format for pycocoevalcap
    # res = {img_id: [{'caption': c}]}
    # gts = {img_id: [{'caption': c}]}
    
    res = {}
    gts = {}
    
    # Clean predictions and prepare data
    for i, item in enumerate(data):
        pred = item['prediction']
        if "Assistant:" in pred:
            item['prediction'] = pred.split("Assistant:", 1)[1].strip()
        
        img_id = str(i) # Use index as ID if image path is not unique or complex
        # prediction
        res[img_id] = [{'caption': item['prediction']}]
        # target
        gts[img_id] = [{'caption': item['target']}]

    # Tokenize for COCO metrics
    try:
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        print("Tokenizing captions...")
        tokenizer = PTBTokenizer()
        gts_tokenized = tokenizer.tokenize(gts)
        res_tokenized = tokenizer.tokenize(res)
    except Exception as e:
        print(f"Error during tokenization: {e}")
        print("Skipping COCO metrics due to tokenization failure.")
        gts_tokenized = None
        res_tokenized = None

    metrics = {}

    # --- CIDEr & SPICE ---
    try:
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.spice.spice import Spice
        
        if gts_tokenized is None or res_tokenized is None:
            raise ImportError("Tokenization failed, cannot compute CIDEr/SPICE")

        # CIDEr
        try:
            print("Computing CIDEr...")
            scorer_cider = Cider()
            score_cider, _ = scorer_cider.compute_score(gts_tokenized, res_tokenized)
            metrics['CIDEr'] = score_cider
            print(f"CIDEr: {score_cider}")
        except Exception as e:
            print(f"Error computing CIDEr: {e}")

        # SPICE
        try:
            print("Computing SPICE...")
            
            from pycocoevalcap.spice.spice import Spice
            print("Computing SPICE (this may take a while)...")
            
            # SPICE requires Java - check if available
            import subprocess
            try:
                subprocess.run(['java', '-version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise RuntimeError("Java not found. SPICE requires Java to be installed.")
            
            scorer_spice = Spice()
            score_spice, _ = scorer_spice.compute_score(gts_tokenized, res_tokenized)
            metrics['SPICE'] = float(score_spice)
            print(f"SPICE: {score_spice:.4f}")
        except Exception as e:
            print(f"Error computing SPICE: {e}")
            print("Note: SPICE requires a compatible Java environment (Java 8). Skipping.")
        
    except ImportError:
        print("Warning: pycocoevalcap not installed or failed to import. Skipping CIDEr/SPICE.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error computing CIDEr/SPICE: {e}")

    # --- CLIPScore ---
    try:
        from torchmetrics.multimodal import CLIPScore
        import torch
        from PIL import Image
        
        print("Computing CLIPScore...")
        metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
        
        # Prepare lists
        # CLIPScore needs (images, text)
        # inference_results.json has "image_path"
        
        predictions = [item['prediction'] for item in data]
        image_paths = [item['image_path'] for item in data]
        
        # We process in batches to avoid OOM
        clip_scores = []
        batch_size = 32
        
        for i, range_start in enumerate(range(0, len(data), batch_size)):
            print(f"Processing CLIPScore batch {i + 1}/{(len(data)-1)//batch_size + 1}...", end='\r')
            batch_preds = predictions[range_start:range_start+batch_size]
            batch_paths = image_paths[range_start:range_start+batch_size]
            
            # Load images
            images = []
            valid_preds = []
            for p, path in zip(batch_preds, batch_paths):
                try:
                    full_path = os.path.join(image_root, path)
                    img = Image.open(full_path).convert("RGB")
                    # Convert to tensor (0-255 scale usually required by torchmetrics, handled internally?)
                    # torchmetrics CLIPScore expects int tensor or PIL
                    import torchvision.transforms.functional as F
                    # img_tensor = F.pil_to_tensor(img) # might be needed
                    images.append(img) # simple list of PIL images is supported validation
                    valid_preds.append(p)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    
            if not images:
                continue
                

            

            
            import torchvision.transforms as T
            transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
            
            imgs_tensor = torch.stack([transform(img) * 255 for img in images]).to(dtype=torch.uint8)
            
            score = metric(imgs_tensor, valid_preds)
            # ClipScore returns a scalar tensor
        
        
        final_clip_score = metric.compute().item()
        metrics['CLIPScore'] = final_clip_score
        print(f"CLIPScore: {final_clip_score}")
        
    except ImportError:
        print("Warning: torchmetrics or torch not installed. Skipping CLIPScore.")
    except Exception as e:
        print(f"Error computing CLIPScore: {e}")

    # Save metrics
    output_file = "evaluation_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="inference_results.json", help="Path to inference results")
    parser.add_argument("--image_root", type=str, default=".")
    args = parser.parse_args()
    
    evaluate_vlm(args.results, args.image_root)