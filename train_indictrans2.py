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
    base_path = "/Users/striker/Desktop/Exploratory/bengali_hindi"
    dataset_path = os.path.join(base_path, "hf_dataset")
    
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset not found at {dataset_path}. Please run prepare_data.py first.")
        
    dataset = load_from_disk(dataset_path)
    
    # Model configuration for IndicTrans2
    model_checkpoint = "ai4bharat/indictrans2-indic-indic-dist-320M"
    src_lang = "hin_Deva"
    tgt_lang = "ben_Beng"
    
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
        inputs = [ex["hi"] for ex in examples["translation"]]
        targets = [ex["bn"] for ex in examples["translation"]]
        
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
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    output_dir = os.path.join(base_path, "indictrans2_finetuned_hi_bn")
    
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
        report_to="none"
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
