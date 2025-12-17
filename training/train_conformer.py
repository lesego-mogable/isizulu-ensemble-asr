import os
import torch
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2ConformerForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
)
from src.utils import remove_special_characters, filter_duration
import evaluate
import numpy as np

# --- Configuration ---
DATASET_NAME = "aconeil/nchlt"
MODEL_NAME = "facebook/wav2vec2-conformer-large-960h" # Or your specific starting checkpoint
OUTPUT_DIR = "./models/conformer_agent"

def compute_metrics(pred):
    wer_metric = evaluate.load("wer")
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

def main():
    # 1. Load Dataset
    dataset = load_dataset(DATASET_NAME, "default", split="train")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # 2. Preprocess
    dataset = dataset.map(remove_special_characters)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.filter(filter_duration)

    # 3. Load Processor & Model
    global processor # Make global for compute_metrics
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ConformerForCTC.from_pretrained(
        MODEL_NAME, 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    model.freeze_feature_encoder()

    # 4. Prepare Data
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(audio["array"], sampling_rate=16000).input_values[0]
        batch["labels"] = processor(text=batch["transcription"]).input_ids
        return batch

    processed_dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names)

    # 5. Data Collator
    from transformers import DataCollatorCTCWithPadding
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # 6. Trainer
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        group_by_length=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=10,
        fp16=torch.cuda.is_available(),
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=3e-4,
        warmup_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()