import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pyloudnorm as pyln
import torchaudio.transforms as T
import random



# def normalize_audio(waveform, sampling_rate=16000):
#     data = waveform.numpy().T # switch channel and data dimension

#     # measure the loudness first 
#     meter = pyln.Meter(sampling_rate) # create BS.1770 meter
#     loudness = meter.integrated_loudness(data)

#     # loudness normalize audio to -24 dB LUFS
#     loudness_normalized_audio = torch.from_numpy(pyln.normalize.loudness(data, loudness, -24.0)).T # switch channel and data dimension back

#     return loudness_normalized_audio.float()

# class AugmentedDataset(Dataset):
#     """Applies simple augmentation (time masking) on top of base dataset."""
#     def __init__(self, base_dataset):
#         self.base_dataset = base_dataset
#         self.time_masking = T.TimeMasking(time_mask_param=10)

#     def __len__(self):
#         return len(self.base_dataset)

#     def __getitem__(self, idx):
#         data, target = self.base_dataset[idx]
#         return self.time_masking(data), target

# class AugmentedDataset(Dataset):
#     def __init__(self, base_dataset):
#         self.base_dataset = base_dataset
#         self.time_mask = T.TimeMasking(time_mask_param=10)
#         self.cache = {}  # 🔹 cache to store preprocessed samples

#     def __len__(self):
#         return len(self.base_dataset)

#     def __getitem__(self, idx):
#         # 🔹 Check if already processed
#         if idx not in self.cache:
#             data, target = self.base_dataset[idx]

#             # 🔹 Normalize only once and store in cache
#             data = data / (data.abs().max() + 1e-8)
#             target = target / (target.abs().max() + 1e-8)
#             self.cache[idx] = (data, target)

#         data, target = self.cache[idx]

#         # 🔹 Apply augmentations on-the-fly
#         noisy = self.time_mask(data)
        
#         return noisy, target

class AugmentedDataset(Dataset):
    """Augments noisy-clean audio pairs with realistic waveform transformations."""
    def __init__(self, base_dataset, sampling_rate=16000):
        self.base_dataset = base_dataset
        self.sr = sampling_rate
        self.time_mask = T.TimeMasking(time_mask_param=10)
        # self.freq_mask = T.FrequencyMasking(freq_mask_param=8)
        # self.pitch_shift = T.PitchShift(self.sr, n_steps=2)
        # self.reverb = T.Vol(0.5)
        # self.noise_types = ["gaussian", "pink", "brown"]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data, target = self.base_dataset[idx]
        
        
        noisy = self.time_mask(data)
        
        return noisy, target
        #noisy  = normalize_audio(noisy)
        #target = normalize_audio(target)
        
        

        # # 1️⃣ Random gain (volume scaling)
        # if random.random() < 0.5:
        #     gain = torch.exp(torch.randn(1) * 0.1)  # ±10 %
        #     noisy *= gain
        #     target *= gain

        # # 2️⃣ Random additive noise
        # if random.random() < 0.6:
        #     std = random.uniform(0.005, 0.05)
        #     noise = torch.randn_like(noisy) * std
        #     noisy = noisy + noise

        # # 3️⃣ Random time masking
        # if random.random() < 0.3:
        #     noisy = self.time_mask(noisy)
            
        # # 4️⃣ Random polarity inversion
        # if random.random() < 0.1:
        #     noisy = -noisy
        # 5 Normalize both signals to -24 LUFS

        # # ⃣ Random frequency masking (applied in mel domain)
        # if random.random() < 0.3:
        #     mel = T.MelSpectrogram(sample_rate=self.sr)(noisy)
        #     mel = self.freq_mask(mel)
        #     noisy = T.InverseMelScale(n_stft=mel.size(1))(mel)   DIDNT WORK

        # # 6️⃣ Random clipping (simulates mic distortion)
        # if random.random() < 0.2:
        #     clip_val = random.uniform(0.5, 1.0)
        #     noisy = torch.clamp(noisy, -clip_val, clip_val)

        # # 7️⃣ Random reverb simulation (gain mod)
        # if random.random() < 0.2:
        #     noisy = self.reverb(noisy)

        # # 8️⃣ Random short-time energy drop (dropout)
        # if random.random() < 0.3:
        #     start = random.randint(0, noisy.size(-1) - self.sr // 10)
        #     length = random.randint(self.sr // 200, self.sr // 50)
        #     noisy[:, start:start + length] *= 0

        # # 🔟 Optional pre-emphasis (can help clarity)
        # if random.random() < 0.5:
        #     noisy = torch.cat(
        #         [noisy[:, :1], noisy[:, 1:] - 0.97 * noisy[:, :-1]], dim=-1
        #     )

        
      