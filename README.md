# Bengali to Hindi Machine Translation fine-tuning

This repository contains fully self-contained standard Python scripts designed to run perfectly on Lightning AI GPU studios (and any standard PyTorch GPU environment).

## Setup
First, ensure you have the required libraries installed:
```bash
pip install -r requirements.txt
```

## Pipeline Execution

### 1. Data Preparation
The dataset preparation script iterates through all matched domains in the `Bengali` and `Hindi` folders to create parallel sentence pairs. It automatically removes any blank lines and mismatched files, converting everything to a unified Hugging Face dataset.
To run the split (80% training / 20% testing):
```bash
python prepare_data.py
```
*This will generate a `hf_dataset` folder containing the train/test splits.*

### 2. Fine-tuning IndicTrans2 Model
To fine-tune the `ai4bharat/indictrans2-indic-indic-dist-320M` model:
```bash
python train_indictrans2.py
```
*This handles tokenization, training, memory usage optimization natively via HF Trainer, and executes a full testing evaluation step automatically after the training is concluded.*

### 3. Fine-tuning NLLB Model (600M Distilled)
To fine-tune the `facebook/nllb-200-distilled-600M` model:
```bash
python train_nllb.py
```
*This similarly leverages your local datasets, assigns correct specific language metrics (ben_Beng & hin_Deva), and outputs a completely evaluated model checkpoint ready for downstream use.*

## Outputs
- **Dataset:** `hf_dataset/`
- **IndicTrans2 Model Checkpoint:** `indictrans2_finetuned_bn_hi/`
- **NLLB Model Checkpoint:** `nllb_finetuned_bn_hi/`