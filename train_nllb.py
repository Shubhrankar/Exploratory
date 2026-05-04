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

def main():
    base_path = "./"
    dataset_path = os.path.join(base_path, "hf_dataset")
    
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset not found at {dataset_path}. Please run prepare_data.py first.")
        
    dataset = load_from_disk(dataset_path)
    
    # Model configuration for NLLB
    model_checkpoint = "facebook/nllb-200-distilled-600M"
    src_lang = "ben_Beng"
    tgt_lang = "hin_Deva"
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, src_lang=src_lang, tgt_lang=tgt_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    
    max_length = 128
    
    def preprocess_function(examples):
        inputs = [ex["bn"] for ex in examples["translation"]]
        targets = [ex["hi"] for ex in examples["translation"]]
        
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
    # generating without the proper target language constraints (forced_bos_token_id).
    # We use predict_with_generate=False for faster training and will evaluate 
    # comprehensively using evaluate_models.py after training completes.

    output_dir = os.path.join(base_path, "nllb_finetuned_bn_hi")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=False, # Disabled for parity with IndicTrans2 and avoiding bad decoding
        fp16=torch.cuda.is_available(), # Use FP16 if running on GPU
        push_to_hub=False,
        report_to="none" # Set to "wandb" if you use weights and biases
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
    
    print("Starting training NLLB...")
    trainer.train()
    
    print("Evaluating NLLB...")
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)
    
    print("Saving the model...")
    trainer.save_model(os.path.join(output_dir, "final_model"))
    print("Done!")

if __name__ == "__main__":
    main()
