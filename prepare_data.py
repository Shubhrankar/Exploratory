import os
from pathlib import Path
from datasets import Dataset, DatasetDict

def load_data(base_path):
    hindi_dir = Path(base_path) / "Hindi"
    bengali_dir = Path(base_path) / "Bengali"
    
    data = []
    
    # Iterate through all domains (subdirectories)
    for domain_dir in hindi_dir.iterdir():
        if not domain_dir.is_dir():
            continue
            
        domain = domain_dir.name
        bn_domain_dir = bengali_dir / domain
        
        if not bn_domain_dir.exists():
            print(f"Warning: Corresponding Bengali domain directory not found for {domain}")
            continue
            
        for hi_file in domain_dir.glob("*.txt"):
            bn_file = bn_domain_dir / hi_file.name
            
            if not bn_file.exists():
                print(f"Warning: Bengali file not found for {hi_file.name}")
                continue
                
            with open(hi_file, 'r', encoding='utf-8') as f_hi, open(bn_file, 'r', encoding='utf-8') as f_bn:
                hi_lines = [line.strip() for line in f_hi.readlines()]
                bn_lines = [line.strip() for line in f_bn.readlines()]
                
                if len(hi_lines) != len(bn_lines):
                    print(f"Warning: Line count mismatch in {hi_file.name}. Hindi: {len(hi_lines)}, Bengali: {len(bn_lines)}. Skipping...")
                    continue
                    
                for hi_text, bn_text in zip(hi_lines, bn_lines):
                    if hi_text and bn_text: # Skip empty lines
                        data.append({
                            "translation": {
                                "hi": hi_text,
                                "bn": bn_text
                            }
                        })
                        
    return data

if __name__ == "__main__":
    base_path = "/Users/striker/Desktop/Exploratory/bengali_hindi"
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
