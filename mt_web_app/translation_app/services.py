import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from comet import download_model, load_from_checkpoint
try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    pass

class TranslationEnsemble:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TranslationEnsemble, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self):
        if self.initialized: return
        print("Initializing Machine Translation Ensemble...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.nllb_model_path = os.path.join(base_path, "nllb_finetuned_bn_hi", "final_model")
        if not os.path.exists(self.nllb_model_path): self.nllb_model_path = "facebook/nllb-200-distilled-600M"
        
        self.indic_model_path = os.path.join(base_path, "indictrans2_finetuned_bn_hi", "final_model")
        if not os.path.exists(self.indic_model_path): self.indic_model_path = "ai4bharat/indictrans2-indic-indic-dist-320M"

        print("Loading NLLB...")
        self.nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="ben_Beng", fix_mistral_regex=True)
        self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(self.nllb_model_path).to(self.device)
        self.nllb_model.eval()
        self.nllb_target_id = self.nllb_tokenizer.lang_code_to_id["hin_Deva"] if hasattr(self.nllb_tokenizer, "lang_code_to_id") else self.nllb_tokenizer.convert_tokens_to_ids("hin_Deva")

        print("Loading IndicTrans2...")
        self.indic_ip = None
        try: self.indic_ip = IndicProcessor(inference=True)
        except NameError: pass
        self.indic_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-indic-indic-dist-320M", trust_remote_code=True)
        self.indic_model = AutoModelForSeq2SeqLM.from_pretrained(self.indic_model_path, trust_remote_code=True).to(self.device)
        self.indic_model.eval()

        print("Loading COMET-Kiwi...")
        comet_path = download_model("Unbabel/wmt22-cometkiwi-da")
        self.qe_model = load_from_checkpoint(comet_path)
        self.initialized = True
        print("Initialization complete.")

    def translate_nllb(self, text):
        inputs = self.nllb_tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            generated_tokens = self.nllb_model.generate(**inputs, max_length=128, forced_bos_token_id=self.nllb_target_id)
        return self.nllb_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()

    def translate_indic(self, text):
        processed_text = [text]
        if self.indic_ip is not None:
            processed_text = self.indic_ip.preprocess_batch([text], src_lang="ben_Beng", tgt_lang="hin_Deva")
        inputs = self.indic_tokenizer(processed_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            generated_tokens = self.indic_model.generate(**inputs, max_length=128, use_cache=False)
        decoded = self.indic_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        if self.indic_ip is not None:
            decoded = self.indic_ip.postprocess_batch(decoded, lang="hin_Deva")
        return decoded[0].strip()

    def calculate_reference_metrics(self, prediction, reference):
        import evaluate
        sacrebleu = evaluate.load("sacrebleu")
        meteor = evaluate.load("meteor")
        bertscore = evaluate.load("bertscore")
        
        bleu = sacrebleu.compute(predictions=[prediction], references=[[reference]])["score"]
        met = meteor.compute(predictions=[prediction], references=[reference])["meteor"]
        bert = bertscore.compute(predictions=[prediction], references=[reference], lang="hi")["f1"][0]
        return {"bleu": bleu, "meteor": met, "bertscore": bert}

    def translate(self, text, model_choice="ensemble", reference=""):
        if not self.initialized: self.initialize()
        
        if model_choice == "nllb":
            translation, model_name, score = self.translate_nllb(text), "NLLB-600M", 0.0
        elif model_choice == "indic":
            translation, model_name, score = self.translate_indic(text), "IndicTrans2-320M", 0.0
        else:
            nllb_out = self.translate_nllb(text)
            indic_out = self.translate_indic(text)
            qe_data = [{"src": text, "mt": nllb_out}, {"src": text, "mt": indic_out}]
            results = self.qe_model.predict(qe_data, batch_size=2, gpus=1 if self.device=="cuda" else 0)
            if results.scores[0] > results.scores[1]: 
                translation, model_name, score = nllb_out, "NLLB-600M", results.scores[0]
            else: 
                translation, model_name, score = indic_out, "IndicTrans2-320M", results.scores[1]
                
        metrics_dict = None
        if reference:
            metrics_dict = self.calculate_reference_metrics(translation, reference)
            
        return translation, model_name, score, metrics_dict

ensemble = TranslationEnsemble()
