import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ConformerForCTC
from safetensors.torch import load_file
import torchaudio
from tqdm import tqdm
import os
import re

# Import Architectures from src
from src.architectures import ASRModel, FusionModel
from src.utils import remove_special_characters

# --- Config ---
CONFORMER_PATH = "./models/conformer_agent"
WAV2VEC2_PATH = "./models/wav2vec2_agent"
CUSTOM_MODEL_PATH = "./models/custom_cnn_rnn"
OUTPUT_DIR = "./models/fusion_model"
BATCH_SIZE = 4
EPOCHS = 5

class EnsembleLogitsDataset(Dataset):
    def __init__(self, hf_dataset, processor, conformer, w2v2, custom, device):
        self.dataset = hf_dataset
        self.processor = processor
        self.conformer = conformer
        self.w2v2 = w2v2
        self.custom = custom
        self.device = device
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80).to(device)

    def __len__(self):
        return len(self.dataset)

    @torch.no_grad()
    def __getitem__(self, idx):
        item = self.dataset[idx]
        waveform = torch.tensor(item["audio"]["array"], dtype=torch.float32)
        
        # 1. Base Agents Inference
        input_values = self.processor(waveform, return_tensors="pt", sampling_rate=16000).input_values.to(self.device)
        
        l_conf = self.conformer(input_values).logits
        l_w2v2 = self.w2v2(input_values).logits
        
        # Custom model processing
        log_mel = torch.log(torch.clamp(self.mel_transform(waveform.to(self.device)), min=1e-5))
        l_custom = self.custom(log_mel.unsqueeze(0))['logits']

        # 2. Align Lengths
        min_len = min(l_conf.shape[1], l_w2v2.shape[1], l_custom.shape[1])
        cat_logits = torch.cat([
            l_conf[:, :min_len, :], 
            l_w2v2[:, :min_len, :], 
            l_custom[:, :min_len, :]
        ], dim=-1).squeeze(0) # (Seq_Len, Combined_Features)

        labels = self.processor.tokenizer(item["transcription"]).input_ids
        
        return {"inputs": cat_logits, "labels": labels}

def collate_fn(batch):
    inputs = [item['inputs'] for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    
    # Pad inputs (Time dimension is dim 0)
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    
    # Flatten labels for CTC
    labels_concat = torch.cat(labels)
    
    input_lens = torch.tensor([x.shape[0] for x in inputs])
    target_lens = torch.tensor([len(x) for x in labels])
    
    return inputs_padded, labels_concat, input_lens, target_lens

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Models
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_PATH)
    
    conformer = Wav2Vec2ConformerForCTC.from_pretrained(CONFORMER_PATH).to(device).eval()
    w2v2 = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_PATH).to(device).eval()
    
    custom = ASRModel(n_mels=80, rnn_dim=512, vocab_size=len(processor.tokenizer), n_cnn_layers=3, n_rnn_layers=3, dropout=0.1)
    custom.load_state_dict(load_file(os.path.join(CUSTOM_MODEL_PATH, "model.safetensors")))
    custom.to(device).eval()

    # Data
    ds = load_dataset("aconeil/nchlt", split="train").map(remove_special_characters)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    # Filter strictly here
    ds = ds.filter(lambda x: x['transcription'] != "")

    train_ds = EnsembleLogitsDataset(ds, processor, conformer, w2v2, custom, device)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0)

    # Fusion Model
    # Input size = vocab_size * 3 (roughly 30-35 * 3 approx 90-100)
    # We dynamically check input size in first iteration usually, but here we assume we know it.
    # Based on your notebook, input was 92.
    fusion_model = FusionModel(input_size=92, hidden_size=512, output_size=len(processor.tokenizer)).to(device)
    
    optimizer = AdamW(fusion_model.parameters(), lr=1e-4)
    ctc_loss = nn.CTCLoss(blank=processor.tokenizer.pad_token_id, zero_infinity=True)

    print("Starting Ensemble Training...")
    fusion_model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for inputs, targets, in_lens, out_lens in tqdm(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = fusion_model(inputs)
            
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = ctc_loss(log_probs, targets, in_lens, out_lens)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader)}")
        torch.save(fusion_model.state_dict(), os.path.join(OUTPUT_DIR, "trained_fusion_model.pth"))

if __name__ == "__main__":
    main()