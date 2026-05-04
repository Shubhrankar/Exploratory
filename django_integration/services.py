import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    IndicProcessor = None

class TranslationService:
    _instance = None
    _models = {}
    _metrics = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TranslationService, cls).__new__(cls)
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance.ip = IndicProcessor(inference=True) if IndicProcessor else None
        return cls._instance

    def load_model(self, model_name, model_path, is_nllb=False, tokenizer_path=None):
        if model_name in self._models:
            return  # Already loaded
            
        print(f"Loading {model_name} into memory on {self.device}...")
        try:
            if is_nllb:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, src_lang="hin_Deva", fix_mistral_regex=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=True)
                
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True).to(self.device)
            model.eval()
            if not is_nllb:
                model.config.use_cache = False
                
            self._models[model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "is_nllb": is_nllb
            }
            print(f"{model_name} loaded successfully!")
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")

    def translate(self, text, model_name="nllb", src_lang="hin_Deva", tgt_lang="ben_Beng"):
        if model_name not in self._models:
            return "Error: Model not loaded."
            
        model_data = self._models[model_name]
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        is_nllb = model_data["is_nllb"]

        if not is_nllb and self.ip is not None:
            processed_text = self.ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
        else:
            processed_text = [text]

        inputs = tokenizer(processed_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        generation_kwargs = {"max_length": 128}

        if is_nllb:
            if hasattr(tokenizer, "lang_code_to_id"):
                forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
            else:
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
            generation_kwargs["forced_bos_token_id"] = forced_bos_token_id
        else:
            generation_kwargs["use_cache"] = False

        with torch.no_grad():
            generated_tokens = model.generate(**inputs, **generation_kwargs)

        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        if not is_nllb and self.ip is not None:
            decoded = self.ip.postprocess_batch(decoded, lang=tgt_lang)

        return decoded[0]

    def get_metrics(self, prediction, reference, source_text):
        if not self._metrics:
            print("Loading evaluation metrics into memory...")
            import evaluate
            self._metrics['bleu'] = evaluate.load("sacrebleu")
            self._metrics['meteor'] = evaluate.load("meteor")
            self._metrics['bertscore'] = evaluate.load("bertscore")
            self._metrics['comet'] = evaluate.load("comet")
        
        bleu = self._metrics['bleu'].compute(predictions=[prediction], references=[reference])
        met = self._metrics['meteor'].compute(predictions=[prediction], references=[reference])
        bert = self._metrics['bertscore'].compute(predictions=[prediction], references=[reference], lang="bn")
        comet_s = self._metrics['comet'].compute(predictions=[prediction], references=[reference], sources=[source_text])

        return {
            'bleu': round(bleu['score'], 2),
            'meteor': round(met['meteor'], 4),
            'bertscore': round(bert['f1'][0], 4),
            'comet': round(comet_s['mean_score'], 4)
        }


# Initialize Singleton instance
translator = TranslationService()

# Automatically load models (adjust paths as needed for your Lightning AI environment)
BASE_PATH = "./"
NLLB_PATH = os.path.join(BASE_PATH, "nllb_finetuned_hi_bn", "final_model")
INDIC_PATH = os.path.join(BASE_PATH, "indictrans2_finetuned_hi_bn", "final_model")

translator.load_model(
    "nllb", 
    model_path=NLLB_PATH if os.path.exists(NLLB_PATH) else "facebook/nllb-200-distilled-600M",
    tokenizer_path="facebook/nllb-200-distilled-600M",
    is_nllb=True
)

translator.load_model(
    "indictrans2", 
    model_path=INDIC_PATH if os.path.exists(INDIC_PATH) else "ai4bharat/indictrans2-indic-indic-dist-320M",
    tokenizer_path="ai4bharat/indictrans2-indic-indic-dist-320M",
    is_nllb=False
)
