import librosa
from pathlib import Path

import torch
from speechbrain.lobes.features import Fbank

utt_id = 'ERMS_a0487'.split('_')
wav_path = Path(f'datasets/L2_ARCTIC/original_dataset/{utt_id[0]}/wav/arctic_{utt_id[1]}.wav')

sample_rate = 16000
hop_length = 20  # ms
n_fft = 400
n_mels = 40

y, _ = librosa.load(wav_path, sr=sample_rate)

y_batch = torch.unsqueeze(torch.from_numpy(y), dim=0)
sb_feat = Fbank(sample_rate=sample_rate, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels)(y_batch)
sb_feat = torch.squeeze(sb_feat)

hop_length_frame = int(hop_length / 1000 * sample_rate)
lib_feat = librosa.feature.melspectrogram(y, sr=sample_rate, hop_length=hop_length_frame, n_fft=n_fft, n_mels=n_mels).T


print(sb_feat.shape)
print(lib_feat.shape)
print(y.shape[0] / hop_length_frame)
