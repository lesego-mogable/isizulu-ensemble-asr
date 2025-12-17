import torch
import torchaudio
import os
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ConformerForCTC
from safetensors.torch import load_file
from src.architectures import ASRModel, FusionModel

class InferenceEnsemble:
    def __init__(self, conformer_path, wav2vec2_path, custom_model_path, fusion_model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing ensemble on {self.device}...")

        # Load Processor (assuming they share the same vocab)
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec2_path)
        
        # 1. Load Conformer Agent
        self.conformer_model = Wav2Vec2ConformerForCTC.from_pretrained(conformer_path).to(self.device).eval()
        
        # 2. Load Wav2Vec2 Agent
        self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_path).to(self.device).eval()

        # 3. Load Custom CNN-RNN Agent
        self.custom_model = ASRModel(n_mels=80, rnn_dim=512, vocab_size=29, n_cnn_layers=3, n_rnn_layers=3, dropout=0.1)
        # Note: Adjust path handling for deployment vs local
        weights_path = os.path.join(custom_model_path, "model.safetensors")
        self.custom_model.load_state_dict(load_file(weights_path, device="cpu"))
        self.custom_model.to(self.device).eval()

        # 4. Load Fusion Model
        # Input size hardcoded to 92 based on your notebook analysis
        self.fusion_model = FusionModel(
            input_size=92, 
            hidden_size=512, 
            output_size=self.processor.tokenizer.pad_token_id + 1
        ).to(self.device)
        self.fusion_model.load_state_dict(torch.load(fusion_model_path, map_location=self.device))
        self.fusion_model.eval()
        
        # Audio Transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=80
        ).to(self.device)

        print("✅ Ensemble System Loaded")

    @torch.no_grad()
    def predict(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

        # Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        waveform = waveform.squeeze(0)

        # Process Base Agents
        input_values = self.processor(waveform.numpy(), return_tensors="pt", sampling_rate=16000).input_values.to(self.device)
        
        logits_conf = self.conformer_model(input_values).logits
        logits_w2v2 = self.wav2vec2_model(input_values).logits
        
        mel_spec = self.mel_transform(waveform.to(self.device))
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        logits_custom = self.custom_model(log_mel_spec.unsqueeze(0))['logits']

        # Align lengths
        min_len = min(logits_conf.shape[1], logits_w2v2.shape[1], logits_custom.shape[1])
        logits_conf = logits_conf[:, :min_len, :]
        logits_w2v2 = logits_w2v2[:, :min_len, :]
        logits_custom = logits_custom[:, :min_len, :]

        # Fusion
        concat_logits = torch.cat([logits_conf, logits_w2v2, logits_custom], dim=-1)
        final_logits = self.fusion_model(concat_logits)

        # Decode
        pred_ids = torch.argmax(final_logits, dim=-1)
        transcription = self.processor.batch_decode(pred_ids)[0]

        return transcription