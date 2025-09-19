
# Whisper Fine-Tuning for Urdu & Pashto

This repository contains scripts to fine-tune [OpenAI Whisper](https://github.com/openai/whisper) for **Urdu** and **Pashto**, languages not officially supported in the original Whisper model.  
By training on Common Voice and custom datasets, these models achieve improved transcription accuracy for low-resource languages.


## 🚀 Features
- Fine-tunes Whisper (tiny/small) on Urdu and Pashto datasets
- On-the-fly dataset preparation from `.tsv` and `.wav` files
- Uses Hugging Face `transformers` + `Seq2SeqTrainer`
- Outputs language-specific fine-tuned Whisper checkpoints

---

## 📂 Repository Structure
```

whisper-finetune-urdu-pashto/
│── finetune\_whisper\_urdu.py     # Fine-tuning for Urdu
│── finetune\_whisper\_pashto.py   # Fine-tuning for Pashto
│── requirements.txt             # Dependencies
│── README.md                    # Project documentation

````

---

## ⚙️ Installation
Install dependencies:
```bash
pip install -r requirements.txt
````

---

## 🏋️ Training

### Urdu Fine-Tuning

```bash
python finetune_whisper_urdu.py
```

### Pashto Fine-Tuning

```bash
python finetune_whisper_pashto.py
```

Both scripts will:

1. Preprocess dataset (`.tsv` → `.json`)
2. Initialize Whisper + tokenizer
3. Train using Hugging Face `Seq2SeqTrainer`
4. Save model & processor to the specified output directory

---

## 📊 Datasets

* [Mozilla Common Voice](https://commonvoice.mozilla.org/) (Urdu, Pashto subsets)
* Additional curated `.wav` + `.tsv` datasets

---

## 📦 Output

After training, models will be saved to:

```
/content/drive/MyDrive/urdu_whisper_tiny
/content/drive/MyDrive/pashto_whisper_tiny
```

You can load them back with:

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("path/to/model")
model = WhisperForConditionalGeneration.from_pretrained("path/to/model")
```

---

## ✅ Requirements

See [requirements.txt](./requirements.txt) for details.

---

## 📌 Notes

* Adjust `MODEL_NAME` to `"openai/whisper-small"` or larger models if GPU allows
* For low-resource environments, reduce batch size or use gradient accumulation
* Training can be resumed by pointing `output_dir` to an existing checkpoint

---


