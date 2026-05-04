import os
from pathlib import Path
from datasets import Dataset, DatasetDict

def load_data(base_path):
    bengali_dir = Path(base_path) / "Bengali"
    hindi_dir = Path(base_path) / "Hindi"
    
    data = []
    
    # Iterate through all domains (subdirectories)
    for domain_dir in bengali_dir.iterdir():
        if not domain_dir.is_dir():
            continue
            
        domain = domain_dir.name
        hi_domain_dir = hindi_dir / domain
        
        if not hi_domain_dir.exists():
            print(f"Warning: Corresponding Hindi domain directory not found for {domain}")
            continue
            
        for bn_file in domain_dir.glob("*.txt"):
            hi_file = hi_domain_dir / bn_file.name
            
            if not hi_file.exists():
                print(f"Warning: Hindi file not found for {bn_file.name}")
                continue
                
            with open(bn_file, 'r', encoding='utf-8') as f_bn, open(hi_file, 'r', encoding='utf-8') as f_hi:
                bn_lines = [line.strip() for line in f_bn.readlines()]
                hi_lines = [line.strip() for line in f_hi.readlines()]
                
                if len(bn_lines) != len(hi_lines):
                    print(f"Warning: Line count mismatch in {bn_file.name}. Bengali: {len(bn_lines)}, Hindi: {len(hi_lines)}. Skipping...")
                    continue
                    
                for bn_text, hi_text in zip(bn_lines, hi_lines):
                    if bn_text and hi_text: # Skip empty lines
                        data.append({
                            "translation": {
                                "bn": bn_text,
                                "hi": hi_text
                            }
                        })
                        
    return data

if __name__ == "__main__":
    base_path = "./"
    print("Loading data from:", base_path)
    data = load_data(base_path)
    print(f"Loaded {len(data)} parallel sentence pairs.")
    
    # Create HuggingFace Dataset
    hf_dataset = Dataset.from_list(data)
    
    # Split 80/20 train/test
    split_dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)
    
    print(f"Train size: {len(split_dataset['train'])}")
    print(f"Test size: {len(split_dataset['test'])}")
    
    # Save to disk
    output_dir = os.path.join(base_path, "hf_dataset")
    split_dataset.save_to_disk(output_dir)
    print(f"Saved dataset to: {output_dir}")
