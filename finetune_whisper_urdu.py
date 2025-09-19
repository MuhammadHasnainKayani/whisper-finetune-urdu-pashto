

# ==================== INSTALL ====================
# !pip install transformers librosa soundfile datasets --quiet

# ==================== IMPORTS ====================
import os
import json
import torch
import librosa
import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from torch.nn.utils.rnn import pad_sequence

# ==================== CONFIGURATION ====================
AUDIO_DIR  = "/content/validated_wav_clip"
TSV_PATH   = "/content/validated.tsv"
JSON_PATH  = "/content/urdu_whisper_data.json"
OUTPUT_DIR = "/content/drive/MyDrive/urdu_whisper_tiny"
MODEL_NAME = "openai/whisper-tiny"  # or "openai/whisper-small"
SR         = 16000
BATCH_SIZE = 4       # small batch to limit RAM use
EPOCHS     = 1
LR         = 1e-5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== STEP 1: JSONL Prep ====================
if not os.path.exists(JSON_PATH):
    df = pd.read_csv(TSV_PATH, sep="\t", header=0)
    df = df.sample(frac=1, random_state=42)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            wav_path = os.path.join(AUDIO_DIR, row['path'].replace('.mp3', '.wav'))
            text = str(row['sentence']).strip()
            if os.path.isfile(wav_path) and text:
                f.write(json.dumps({"audio": wav_path, "text": text}, ensure_ascii=False) + "\n")
    print(f"Created JSONL with {len(df)} entries.")
else:
    print("JSONL exists, skipping.")

# ==================== STEP 2: Processor & Model ====================
processor = WhisperProcessor.from_pretrained(
    MODEL_NAME, language="urdu", task="transcribe"
)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# ==================== STEP 3: Dataset + Collator ====================
class WhisperOnTheFlyDataset(Dataset):
    def __init__(self, json_path, sr):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.records = [json.loads(l) for l in f]
        self.sr = sr
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        wav, _ = librosa.load(rec['audio'], sr=self.sr)
        return {"audio": wav, "text": rec['text']}

# On-the-fly collator: returns CPU tensors, Trainer handles device placement

def collate_fn(batch):
    audio_list = [rec['audio'] for rec in batch]
    proc_inputs = processor(audio_list, sampling_rate=SR, return_tensors='pt')
    input_feats = proc_inputs.input_features  # CPU

    texts = [rec['text'].lower() for rec in batch]
    tgt = processor.tokenizer(
        texts,
        padding='longest',
        max_length=128,
        truncation=True,
        return_tensors='pt'
    ).input_ids
    tgt[tgt == processor.tokenizer.pad_token_id] = -100

    return {'input_features': input_feats, 'labels': tgt}

# instantiate dataset
dataset = WhisperOnTheFlyDataset(JSON_PATH, SR)

# ==================== STEP 4: TrainingArguments & Trainer ====================
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
    tokenizer=None,
)

# ==================== STEP 5: Train & Save ====================
trainer.train()
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"✔ Saved model & processor to {OUTPUT_DIR}")
