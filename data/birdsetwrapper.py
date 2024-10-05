from torch.utils.data import Dataset
import torchaudio


class BirdSetWrapper(Dataset):
    def __init__(self, birdset_dataset):
        self.dataset = birdset_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        audio, label = sample["input_values"], sample["labels"]

        # Convert to fbank
        input_value = torchaudio.compliance.kaldi.fbank(audio, htk_compat=True, sample_frequency=16000, use_energy=False,
            window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10).unsqueeze(dim=0)

        return input_value, label