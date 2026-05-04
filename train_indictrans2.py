import os
import torch
import evaluate
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    pass

def main():
    base_path = "./"
    dataset_path = os.path.join(base_path, "hf_dataset")
    
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset not found at {dataset_path}. Please run prepare_data.py first.")
        
    dataset = load_from_disk(dataset_path)
    
    # Model configuration for IndicTrans2
    model_checkpoint = "ai4bharat/indictrans2-indic-indic-dist-320M"
    src_lang = "ben_Beng"
    tgt_lang = "hin_Deva"
    
    print(f"Loading tokenizer and model from {model_checkpoint}...")
    try:
        # Trust remote code in case IndicTrans2 requires custom tokenizer code
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, trust_remote_code=True)
    except Exception as e:
        print(f"\n[ERROR] Failed to load {model_checkpoint}.")
        print("This is likely a Hugging Face authentication issue (403 Forbidden).")
        print("To fix this, please run the following command in your terminal:")
        print("    huggingface-cli login")
        print("And provide a valid token from your Hugging Face account (Settings > Access Tokens) that has read permissions.")
        print(f"\nDetails: {e}")
        return
        
    try:
        ip = IndicProcessor(inference=False)
    except NameError:
        raise ImportError("Please install IndicTransToolkit: pip install git+https://github.com/VarunGumma/IndicTransToolkit.git")
    
    max_length = 128
    
    def preprocess_function(examples):
        inputs = [ex["bn"] for ex in examples["translation"]]
        targets = [ex["hi"] for ex in examples["translation"]]
        
        # Preprocess with IndicProcessor
        inputs = ip.preprocess_batch(inputs, src_lang=src_lang, tgt_lang=tgt_lang)
        # For targets, we also often preprocess to normalize the script
        targets = ip.preprocess_batch(targets, src_lang=tgt_lang, tgt_lang=tgt_lang)
        
        model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
        return model_inputs

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=dataset["train"].column_names
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Evaluation and metric computation during training is disabled to prevent
    # a known generation bug (AttributeError: 'NoneType' object has no attribute 'shape') 
    # specific to IndicTrans2 in HF Trainer. We use predict_with_generate=False and 
    # will evaluate comprehensively using evaluate_models.py after training instead.

    output_dir = os.path.join(base_path, "indictrans2_finetuned_bn_hi")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=False, # Set to False to prevent generation crashes
        fp16=torch.cuda.is_available(), # Use FP16 if running on GPU
        push_to_hub=False,
        report_to="none"
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics omitted as predict_with_generate=False
    )
    
    print("Starting training IndicTrans2...")
    trainer.train()
    
    print("Evaluating IndicTrans2...")
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)
    
    print("Saving the model...")
    trainer.save_model(os.path.join(output_dir, "final_model"))
    print("Done!")

if __name__ == "__main__":
    main()
