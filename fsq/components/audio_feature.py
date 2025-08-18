import torch
import torchaudio.transforms
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


class MFCCEncoder(nn.Module):
    def __init__(self, n_mfcc=32, hop_length=520, transpose=True):
        super().__init__()
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": 2048,
                "n_mels": 256,
                "hop_length": hop_length,  # todo: adjust to n_frames
                "center": False
            },
        )
        self.transpose = transpose

    def forward(self, wav_data):
        feat = self.mfcc_transform(wav_data)
        if self.transpose:
            feat = feat.transpose(1, 2)  # to (batch x seq x dim)
        return feat


class MelSpectrogramEncoder(nn.Module):
    def __init__(self, dim, hop_length=535, transpose=True):
        super().__init__()
        self.spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            hop_length=hop_length,
            n_fft=2048,
            n_mels=dim
        )
        self.transpose = transpose

    def forward(self, wav_data):
        feat = self.spectrogram(wav_data)
        if self.transpose:
            feat = feat.transpose(1, 2)  # to (batch x seq x dim)
        return feat


class Wav2VecEncoder(nn.Module):
    def __init__(self, transpose=True):
        super().__init__()
        model_name = "facebook/wav2vec2-base-960h"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.transpose = transpose

    def forward(self, wav_data):
        i = self.feature_extractor(wav_data, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            o = self.model(i.input_values[0].to(self.model.device))
        feat = o.extract_features

        if self.transpose:
            feat = feat.transpose(1, 2)  # to (batch x dim x seq)
        return feat
