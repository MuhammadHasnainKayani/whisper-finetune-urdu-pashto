# ==================== INSTALL ====================
# !pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 -q
# !pip install datasets transformers librosa soundfile ipython --quiet

# ==================== IMPORTS ====================
import os
import json
import pandas as pd
import torch
import numpy as np
from datasets import Dataset, Audio, load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# ==================== CONFIG ====================
AUDIO_DIR = "/content/ZipValidated/Validated_Clips_wav"
TSV_PATH = "/content/ZipValidated/validated_sorted.tsv"
JSON_PATH = "/content/pashto_whisper_data.json"
OUTPUT_DIR = "/content/drive/MyDrive/whisper_pashto_tiny"
MODEL_NAME = "openai/whisper-base"  # Switched to Base for better capacity

# ==================== STEP 1: TSV → JSONL ====================
if not os.path.exists(JSON_PATH):
    df = pd.read_csv(TSV_PATH, sep='\t', header=None)
    df.columns = ['client_id', 'path', 'sentence_id', 'sentence', 'sentence_domain',
                  'up_votes', 'down_votes', 'age', 'gender', 'accents', 'variant', 'locale', 'segment']
    df = df.sample(frac=0.5, random_state=42)  # Use full data (remove .sample if using full dataset)

    samples = []
    for i, row in df.iterrows():
        audio_path = os.path.join(AUDIO_DIR, row['path'].replace('.mp3', '.wav'))
        if os.path.exists(audio_path) and isinstance(row['sentence'], str) and len(row['sentence']) > 0:
            samples.append({
                "audio": audio_path,
                "text": row['sentence'],
                "language": "ps",
                "id": str(i)
            })

    with open(JSON_PATH, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
else:
    print("JSON exists, skipping conversion.")

# ==================== STEP 2: Dataset with Validation ====================
dataset = load_dataset("json", data_files=JSON_PATH, split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# ==================== STEP 3: Processor with Pashto Checks ====================
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="pashto", task="transcribe")
# Correct Pashto-specific letters
important_pashto_letters = ["ښ", "ګ", "ڼ", "ې", "ۍ", "څ", "ځ", "ږ", "ړ"]

# Tokenization test
sample_text = "په غالیو کې اوږده ګلان ښایسته ښکارېدل، ځینې ښې څانګې ږغېدلې."
tokenized = processor.tokenizer(sample_text).input_ids
decoded = processor.tokenizer.decode(tokenized)

# Check for missing tokens
missing_tokens = []
for char in important_pashto_letters:
    if char not in decoded:
        missing_tokens.append(char)

if missing_tokens:
    processor.tokenizer.add_tokens(missing_tokens)
    print(f"Added tokens: {missing_tokens}")


# ==================== STEP 4: Audio-Text Processing ====================
def prepare_example(batch):
    audio = batch["audio"]
    inputs = processor(audio["array"], sampling_rate=16000)

    # Pashto text cleaning
    text = batch["text"].strip().lower()
    text = text.replace("ـ", "").replace("ۀ", "ه")  # Normalization

    # Tokenize
    labels = processor.tokenizer(
        text,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt"
    ).input_ids[0]
    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {"input_features": inputs["input_features"][0], "labels": labels}

processed_dataset = dataset.map(prepare_example, remove_columns=dataset["train"].column_names)

# ==================== STEP 5: Model Setup ====================
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

if missing_tokens:
    model.resize_token_embeddings(len(processor.tokenizer))

# ==================== STEP 6: Training Arguments ====================
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    num_train_epochs=2,
    save_steps=500,
    fp16=True,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    predict_with_generate=True
)

# ==================== STEP 7: Trainer ====================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["test"],
    tokenizer=processor.feature_extractor,
    data_collator=lambda x: {
        "input_features": torch.stack([torch.tensor(f["input_features"]) for f in x]),
        "labels": torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f["labels"]) for f in x],
            batch_first=True,
            padding_value=-100
        )
    }
)

# ==================== TRAIN ====================
trainer.train()
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
