import os
import torch
import evaluate
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
from comet import download_model, load_from_checkpoint  # Added for QE Combination
from huggingface_hub import login # Added for Hugging Face authentication

# Add your token here - required for gated models like COMET-Kiwi
token = os.getenv("HF_TOKEN")

try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    pass

def generate_predictions(model_path, dataset, src_lang, target_lang, is_nllb=False, batch_size=17, tokenizer_path=None):
    print(f"\n--- Loading model from {model_path} ---")
    
    # Initialize processor for IndicTrans2
    ip = None
    if not is_nllb:
        try:
            ip = IndicProcessor(inference=True)
        except NameError:
            print("Warning: IndicProcessor not defined. Did you install IndicTransToolkit?")

    # Load tokenizers appropriately
    try:
        if is_nllb:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, src_lang=src_lang, fix_mistral_regex=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=True)
            
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"\n[ERROR] Failed to load {model_path}.")
        print("If this is a 403 Forbidden error, please run 'huggingface-cli login' and provide your HF access token.")
        print(f"Details: {e}")
        return [], [], []
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    predictions = []
    references = []
    sources = []
    
    print("Generating predictions...")
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        # Swapped to extract Bengali as source and Hindi as reference
        bengali_texts = [ex["bn"] for ex in batch["translation"]]
        hindi_texts = [ex["hi"] for ex in batch["translation"]]
        
        # Tokenize source
        if not is_nllb and ip is not None:
            processed_bengali_texts = ip.preprocess_batch(bengali_texts, src_lang=src_lang, tgt_lang=target_lang)
        else:
            processed_bengali_texts = bengali_texts
            
        inputs = tokenizer(processed_bengali_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        generation_kwargs = {"max_length": 128}
        if is_nllb:
            # Set target language manually for NLLB generation
            if hasattr(tokenizer, "lang_code_to_id"):
                forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
            else:
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)
            generation_kwargs["forced_bos_token_id"] = forced_bos_token_id
        else:
            generation_kwargs["use_cache"] = False
            
        with torch.no_grad():
            generated_tokens = model.generate(**inputs, **generation_kwargs)
            
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        if not is_nllb and ip is not None:
            decoded_preds = ip.postprocess_batch(decoded_preds, lang=target_lang)
            
        predictions.extend([pred.strip() for pred in decoded_preds])
        references.extend([[ref.strip()] for ref in hindi_texts])
        sources.extend([src.strip() for src in bengali_texts])
        
    return predictions, references, sources

def calculate_metrics(predictions, references, sources, model_name):
    print(f"\nEvaluating performance for {model_name}...")
    
    if not predictions:
        print("No predictions to evaluate (model might have failed to load).")
        return
        
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
    # Swapped lang="bn" to lang="hi" for Hindi evaluation
    bert_results = bertscore.compute(predictions=predictions, references=flat_references, lang="hi")
    print(f"BERTScore (F1 mean): {sum(bert_results['f1'])/len(bert_results['f1']):.4f}")
    
    # Calculate COMET (Requires source sentences)
    comet_results = comet.compute(predictions=predictions, references=flat_references, sources=sources)
    print(f"COMET Score (mean): {comet_results['mean_score']:.4f}")

    print("\n-------------------------------------------------")
    return {
        "bleu": bleu_results['score'],
        "meteor": meteor_results['meteor'],
        "bertscore": sum(bert_results['f1'])/len(bert_results['f1']),
        "comet": comet_results['mean_score']
    }

def combine_predictions_with_qe(sources, nllb_preds, indic_preds):
    print("\n--- Combining outputs using COMET-Kiwi (Quality Estimation) ---")
    
    # Download and load the reference-free QE model
    print("Loading COMET-Kiwi model...")
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    qe_model = load_from_checkpoint(model_path)
    
    # Prepare data format for COMET
    nllb_data = [{"src": src, "mt": mt} for src, mt in zip(sources, nllb_preds)]
    indic_data = [{"src": src, "mt": mt} for src, mt in zip(sources, indic_preds)]
    
    # Score both sets of predictions (this does NOT use the human references)
    print("Scoring NLLB predictions...")
    nllb_results = qe_model.predict(nllb_data, batch_size=8, gpus=1)
    
    print("Scoring IndicTrans2 predictions...")
    indic_results = qe_model.predict(indic_data, batch_size=8, gpus=1)
    
    combined_preds = []
    nllb_wins = 0
    indic_wins = 0
    
    # Iterate through every sentence and pick the highest-scoring translation
    for i in range(len(sources)):
        if nllb_results.scores[i] > indic_results.scores[i]:
            combined_preds.append(nllb_preds[i])
            nllb_wins += 1
        else:
            combined_preds.append(indic_preds[i])
            indic_wins += 1
            
    print(f"\nCombination complete!")
    print(f"NLLB chosen {nllb_wins} times.")
    print(f"IndicTrans2 chosen {indic_wins} times.")
    
    return combined_preds

def main():
    base_path = ""
    dataset_path = os.path.join(base_path, "hf_dataset")
    
    if not os.path.exists(dataset_path):
        raise ValueError("Dataset not found. Run prepare_data.py first.")
        
    # Load the test dataset (20% split)
    dataset = load_from_disk(dataset_path)["test"]
    
    # Define models
    # Fallback to base model if finetuned model directory doesn't exist yet
    nllb_finetuned = os.path.join(base_path, "nllb_finetuned_bn_hi", "final_model")
    nllb_model = nllb_finetuned if os.path.exists(nllb_finetuned) else "facebook/nllb-200-distilled-600M"
    
    indictrans2_finetuned = os.path.join(base_path, "indictrans2_finetuned_bn_hi", "final_model")
    indictrans2_model = indictrans2_finetuned if os.path.exists(indictrans2_finetuned) else "ai4bharat/indictrans2-indic-indic-dist-320M"
    
    # 1. Evaluate NLLB
    nllb_preds, hi_refs, bn_sources = generate_predictions(
        model_path=nllb_model, 
        dataset=dataset, 
        src_lang="ben_Beng", 
        target_lang="hin_Deva",
        is_nllb=True,
        tokenizer_path="facebook/nllb-200-distilled-600M"
    )
    calculate_metrics(nllb_preds, hi_refs, bn_sources, "NLLB-600M")

    # 2. Evaluate IndicTrans2
    indic_preds, _, _ = generate_predictions(
        model_path=indictrans2_model,
        dataset=dataset,
        src_lang="ben_Beng",
        target_lang="hin_Deva",
        is_nllb=False,
        tokenizer_path="ai4bharat/indictrans2-indic-indic-dist-320M"
    )
    calculate_metrics(indic_preds, hi_refs, bn_sources, "IndicTrans2-320M")

    # 3. Combine Outputs Using Quality Estimation
    # (Only runs if both models generated predictions successfully)
    if nllb_preds and indic_preds:
        combined_preds = combine_predictions_with_qe(bn_sources, nllb_preds, indic_preds)

        # 4. Evaluate Final Combined Output
        print("\n=================================================")
        print("Evaluating FINAL COMBINED Output")
        print("=================================================")
        calculate_metrics(combined_preds, hi_refs, bn_sources, "Combined (NLLB + IndicTrans2)")

if __name__ == "__main__":
    main()