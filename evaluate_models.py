import os
import torch
import evaluate
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

def generate_predictions(model_path, dataset, src_lang, target_lang, is_nllb=False, batch_size=16):
    print(f"\n--- Loading model from {model_path} ---")
    
    # Load tokenizers appropriately
    if is_nllb:
        tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang=src_lang)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    predictions = []
    references = []
    sources = []
    
    print("Generating predictions...")
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        hindi_texts = [ex["hi"] for ex in batch["translation"]]
        bengali_texts = [ex["bn"] for ex in batch["translation"]]
        
        # Tokenize source
        inputs = tokenizer(hindi_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        generation_kwargs = {"max_length": 128}
        if is_nllb:
            # Set target language manually for NLLB generation
            forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
            generation_kwargs["forced_bos_token_id"] = forced_bos_token_id
            
        with torch.no_grad():
            generated_tokens = model.generate(**inputs, **generation_kwargs)
            
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        predictions.extend([pred.strip() for pred in decoded_preds])
        references.extend([[ref.strip()] for ref in bengali_texts])
        sources.extend([src.strip() for src in hindi_texts])
        
    return predictions, references, sources

def calculate_metrics(predictions, references, sources, model_name):
    print(f"\nEvaluating performance for {model_name}...")
    
    # Load evaluation metrics
    sacrebleu = evaluate.load("sacrebleu")
    meteor = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore")
    comet = evaluate.load("comet")
    
    # Calculate SacreBLEU
    bleu_results = sacrebleu.compute(predictions=predictions, references=references)
    print(f"BLEU Score: {bleu_results['score']:.2f}")

    # Calculate METEOR
    flat_references = [refs[0] for refs in references] # METEOR typically takes 1D list of refs depending on version
    meteor_results = meteor.compute(predictions=predictions, references=flat_references)
    print(f"METEOR Score: {meteor_results['meteor']:.4f}")
    
    # Calculate BERTScore 
    # (using multilingual bert or muril for Indic languages is best, but evaluate default uses roberta-large / bert-base-multilingual-cased)
    bert_results = bertscore.compute(predictions=predictions, references=flat_references, lang="bn")
    print(f"BERTScore (F1 mean): {sum(bert_results['f1'])/len(bert_results['f1']):.4f}")
    
    # Calculate COMET (Requires source sentences)
    # Using the default model "Unbabel/wmt22-comet-da" 
    comet_results = comet.compute(predictions=predictions, references=flat_references, sources=sources)
    print(f"COMET Score (mean): {comet_results['mean_score']:.4f}")

    print("\n-------------------------------------------------")
    return {
        "bleu": bleu_results['score'],
        "meteor": meteor_results['meteor'],
        "bertscore": sum(bert_results['f1'])/len(bert_results['f1']),
        "comet": comet_results['mean_score']
    }

def main():
    base_path = "/Users/striker/Desktop/Exploratory/bengali_hindi"
    dataset_path = os.path.join(base_path, "hf_dataset")
    
    if not os.path.exists(dataset_path):
        raise ValueError("Dataset not found. Run prepare_data.py first.")
        
    # Load the test dataset (20% split)
    dataset = load_from_disk(dataset_path)["test"]
    
    # Define models
    # Fallback to base model if finetuned model directory doesn't exist yet
    nllb_finetuned = os.path.join(base_path, "nllb_finetuned_hi_bn", "final_model")
    nllb_model = nllb_finetuned if os.path.exists(nllb_finetuned) else "facebook/nllb-200-distilled-600M"
    
    indictrans2_finetuned = os.path.join(base_path, "indictrans2_finetuned_hi_bn", "final_model")
    indictrans2_model = indictrans2_finetuned if os.path.exists(indictrans2_finetuned) else "ai4bharat/indictrans2-indic-indic-dist-320M"
    
    # Evaluate NLLB
    nllb_preds, bn_refs, hi_sources = generate_predictions(
        model_path=nllb_model, 
        dataset=dataset, 
        src_lang="hin_Deva", 
        target_lang="ben_Beng",
        is_nllb=True
    )
    calculate_metrics(nllb_preds, bn_refs, hi_sources, "NLLB-600M")

    # Evaluate IndicTrans2
    indic_preds, _, _ = generate_predictions(
        model_path=indictrans2_model,
        dataset=dataset,
        src_lang="hin_Deva",
        target_lang="ben_Beng",
        is_nllb=False
    )
    calculate_metrics(indic_preds, bn_refs, hi_sources, "IndicTrans2-320M")

if __name__ == "__main__":
    main()
