import csv
import json
import os

def convert_csv_to_json(csv_path, output_dir):
    """
    Converts the road scene captioning CSV to train/val/test JSON files.
    """
    print(f"Reading CSV from {csv_path}...")
    
    splits = {
        "train": [],
        "val": [],
        "test": []
    }
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # CSV columns: image_path, target, split
                item = {
                    "image": row['image_path'],
                    "caption": row['target']
                }
                
                # Map CSV split names to our standard names if necessary
                # CSV seems to use: train, val, test (based on view_file output)
                split_name = row['split'].strip()
                
                if split_name in splits:
                    splits[split_name].append(item)
                else:
                    print(f"Warning: Unknown split '{split_name}' for image {row['image_path']}")

        print(f"Found {len(splits['train'])} training examples.")
        print(f"Found {len(splits['val'])} validation examples.")
        print(f"Found {len(splits['test'])} test examples.")

        # Save to JSON files
        for split_name, data in splits.items():
            output_path = os.path.join(output_dir, f"{split_name}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {output_path}")

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    csv_file = os.path.join("data", "complete_captions_sensation.csv")
    convert_csv_to_json(csv_file, "data")
