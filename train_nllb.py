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
    base_path = "/Users/striker/Desktop/Exploratory/bengali_hindi"
    dataset_path = os.path.join(base_path, "hf_dataset")
    
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset not found at {dataset_path}. Please run prepare_data.py first.")
        
    dataset = load_from_disk(dataset_path)
    
    # Model configuration for NLLB
    model_checkpoint = "facebook/nllb-200-distilled-600M"
    src_lang = "hin_Deva"
    tgt_lang = "ben_Beng"
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    
    max_length = 128
    
    def preprocess_function(examples):
        inputs = [ex["hi"] for ex in examples["translation"]]
        targets = [ex["bn"] for ex in examples["translation"]]
        
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
        
        # Set up the tokenizer for targets
        tokenizer.src_lang = tgt_lang
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_length, truncation=True)
            
        model_inputs["labels"] = labels["input_ids"]
        # Reset source language for next batch
        tokenizer.src_lang = src_lang
        return model_inputs

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=dataset["train"].column_names
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    metric = evaluate.load("sacrebleu")
    
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    output_dir = os.path.join(base_path, "nllb_finetuned_hi_bn")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
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
        compute_metrics=compute_metrics,
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
