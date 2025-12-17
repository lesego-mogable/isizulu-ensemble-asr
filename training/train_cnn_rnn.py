import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset, Audio
import torchaudio
from transformers import Wav2Vec2CTCTokenizer
from src.utils import remove_special_characters, filter_duration
from src.architectures import ASRModel # Importing from your src folder
from tqdm import tqdm

# --- Configuration ---
DATASET_NAME = "aconeil/nchlt"
TOKENIZER_PATH = "./vocab" # Path where you saved your vocab.json
OUTPUT_DIR = "./models/custom_cnn_rnn"
EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-4

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.map(remove_special_characters)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.filter(filter_duration)

    # 2. Tokenizer (Assumes you generated vocab.json previously)
    tokenizer = Wav2Vec2CTCTokenizer(os.path.join(TOKENIZER_PATH, "vocab.json"), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    
    # 3. Transforms
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80).to(device)

    # 4. Collate Function
    def collate_fn(batch):
        inputs = []
        targets = []
        input_lengths = []
        target_lengths = []

        for item in batch:
            waveform = torch.tensor(item["audio"]["array"], dtype=torch.float32).to(device)
            # Log Mel Spectrogram
            spec = torch.log(torch.clamp(mel_transform(waveform), min=1e-5)).squeeze(0).transpose(0, 1) # (Time, Freq)
            inputs.append(spec)
            input_lengths.append(spec.shape[0])
            
            # Tokenize text
            label = torch.tensor(tokenizer(item["transcription"]).input_ids).to(device)
            targets.append(label)
            target_lengths.append(len(label))

        inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True).transpose(1, 2) # (Batch, Freq, Time) -> CNN expects (Batch, C, F, T) handled in model
        targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)
        
        return inputs, targets, torch.tensor(input_lengths), torch.tensor(target_lengths)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # 5. Model
    model = ASRModel(n_mels=80, rnn_dim=512, vocab_size=len(tokenizer), n_cnn_layers=3, n_rnn_layers=3, dropout=0.1).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    ctc_loss = nn.CTCLoss(blank=tokenizer.pad_token_id, zero_infinity=True)

    # 6. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            inputs, targets, input_lens, target_lens = batch
            
            # Adjust input lengths for CNN subsampling (MaxPooling reduces time dim by 2)
            input_lens = input_lens // 2 

            optimizer.zero_grad()
            outputs = model(inputs) # Returns dict {'logits': ...}
            logits = outputs['logits']
            
            # Prepare for CTC: (T, N, C)
            log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
            
            loss = ctc_loss(log_probs, targets, input_lens, target_lens)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader)}")
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    main()