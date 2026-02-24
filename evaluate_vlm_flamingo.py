import json
import argparse
import sys
import os

def evaluate_openflamingo(results_file="inference_results_openflamingo.json", image_root="."):
    print(f"Loading results from {results_file}...")
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {results_file} not found.")
        return

    print(f"Loaded {len(data)} predictions")

    # Prepare data for COCO Eval
    # COCO Eval expects: res = {img_id: [{'caption': c}]}
    #                    gts = {img_id: [{'caption': c}]}
    
    res = {}
    gts = {}
    
    # Clean predictions and prepare data
    for i, item in enumerate(data):
        pred = item['prediction']
        
        # Additional cleaning for common artifacts
        for artifact in ["Assistant:", "A:", "\n\n", "<div", "<ul", "<li"]:
            if artifact in pred:
                pred = pred.split(artifact)[0].strip()
        
        img_id = str(i)
        res[img_id] = [{'caption': pred}]
        gts[img_id] = [{'caption': item['target']}]

    metrics = {}

    # --- CIDEr & SPICE (requires pycocoevalcap) ---
    try:
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        from pycocoevalcap.cider.cider import Cider
        
        print("Tokenizing captions...")
        tokenizer = PTBTokenizer()
        gts_tokenized = tokenizer.tokenize(gts)
        res_tokenized = tokenizer.tokenize(res)

        # CIDEr
        try:
            print("Computing CIDEr...")
            scorer_cider = Cider()
            score_cider, _ = scorer_cider.compute_score(gts_tokenized, res_tokenized)
            metrics['CIDEr'] = float(score_cider)
            print(f"CIDEr: {score_cider:.4f}")
        except Exception as e:
            print(f"Error computing CIDEr: {e}")

        # SPICE (optional - requires Java)
        try:
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
            print(f"Skipping SPICE: {e}")
        
    except ImportError as e:
        print(f"Warning: pycocoevalcap not installed. Skipping CIDEr/SPICE.")
        print("Install with: pip install pycocoevalcap")
    except Exception as e:
        print(f"Error computing CIDEr/SPICE: {e}")

    # --- BLEU, METEOR, ROUGE (using simpler alternatives if pycocoevalcap fails) ---
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        
        if 'gts_tokenized' in locals() and 'res_tokenized' in locals():
            # BLEU
            try:
                print("Computing BLEU...")
                scorer_bleu = Bleu(4)
                bleu_scores, _ = scorer_bleu.compute_score(gts_tokenized, res_tokenized)
                for i, score in enumerate(bleu_scores, 1):
                    metrics[f'BLEU-{i}'] = float(score)
                    print(f"BLEU-{i}: {score:.4f}")
            except Exception as e:
                print(f"Error computing BLEU: {e}")
            
            # METEOR
            try:
                print("Computing METEOR...")
                scorer_meteor = Meteor()
                score_meteor, _ = scorer_meteor.compute_score(gts_tokenized, res_tokenized)
                metrics['METEOR'] = float(score_meteor)
                print(f"METEOR: {score_meteor:.4f}")
            except Exception as e:
                print(f"Error computing METEOR: {e}")
            
            # ROUGE
            try:
                print("Computing ROUGE...")
                scorer_rouge = Rouge()
                score_rouge, _ = scorer_rouge.compute_score(gts_tokenized, res_tokenized)
                metrics['ROUGE-L'] = float(score_rouge)
                print(f"ROUGE-L: {score_rouge:.4f}")
            except Exception as e:
                print(f"Error computing ROUGE: {e}")
    except ImportError:
        print("Skipping BLEU/METEOR/ROUGE (pycocoevalcap not fully installed)")

    # --- CLIPScore ---
    try:
        from torchmetrics.multimodal import CLIPScore
        import torch
        from PIL import Image
        import torchvision.transforms as T
        
        print("Computing CLIPScore...")
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
        
        # CLIPScore needs images and text
        predictions = []
        images_tensor_list = []
        
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        
        batch_size = 16 if device == "cuda" else 8
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batch_preds = []
            batch_imgs = []
            
            for item in batch:
                try:
                    # The key should be 'image' not 'image_path' based on inference output
                    image_path = os.path.join(image_root, item['image'])
                    img = Image.open(image_path).convert("RGB")
                    img_tensor = transform(img)
                    
                    batch_imgs.append(img_tensor)
                    batch_preds.append(item['prediction'])
                except Exception as e:
                    print(f"Warning: Could not load image {item.get('image', 'unknown')}: {e}")
                    continue
            
            if not batch_imgs:
                continue
            
            # Stack images and convert to uint8 (0-255 range)
            imgs_tensor = torch.stack(batch_imgs).to(device)
            imgs_tensor = (imgs_tensor * 255).to(torch.uint8)
            
            # Update metric (accumulates internally)
            metric.update(imgs_tensor, batch_preds)
            
            print(f"Processed {min(i+batch_size, len(data))}/{len(data)} images for CLIPScore...", end='\r')
        
        # Compute final score
        final_clip_score = metric.compute().item()
        metrics['CLIPScore'] = float(final_clip_score)
        print(f"\nCLIPScore: {final_clip_score:.4f}")
        
    except ImportError:
        print("Warning: torchmetrics not installed. Skipping CLIPScore.")
        print("Install with: pip install torchmetrics")
    except Exception as e:
        import traceback
        print(f"Error computing CLIPScore: {e}")
        traceback.print_exc()

    # --- Simple metrics (always computed) ---
    print("\nComputing simple metrics...")
    
    # Average length
    pred_lengths = [len(item['prediction'].split()) for item in data]
    target_lengths = [len(item['target'].split()) for item in data]
    
    metrics['avg_prediction_length'] = sum(pred_lengths) / len(pred_lengths)
    metrics['avg_target_length'] = sum(target_lengths) / len(target_lengths)
    
    print(f"Average prediction length: {metrics['avg_prediction_length']:.2f} words")
    print(f"Average target length: {metrics['avg_target_length']:.2f} words")

    # Save metrics
    output_file = results_file.replace('.json', '_metrics.json')
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print('='*50)
    for metric_name, value in sorted(metrics.items()):
        print(f"{metric_name:25s}: {value:.4f}")
    print('='*50)
    print(f"\nSaved evaluation metrics to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OpenFlamingo inference results")
    parser.add_argument("--results", default="inference_results_openflamingo.json", 
                       help="Path to inference results JSON file")
    parser.add_argument("--image_root", type=str, default=".")
    args = parser.parse_args()
    
    evaluate_openflamingo(args.results, args.image_root)