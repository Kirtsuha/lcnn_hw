from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset


class ASVspoofDataset(Dataset):
    """ASVspoof 2019 Logical Access split described by an official protocol."""

    def __init__(
        self,
        data_dir,
        protocol_path,
        split,
        sample_rate=16000,
        n_fft=512,
        win_length=320,
        hop_length=160,
        n_frames=750,
        random_crop=False,
        limit=None,
        limit_per_class=None,
    ):
        self.data_dir = Path(data_dir).expanduser()
        self.protocol_path = Path(protocol_path).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_frames = n_frames
        self.random_crop = random_crop
        self.window = torch.hann_window(win_length)
        self.index = self._read_protocol()
        if limit_per_class is not None:
            self.index = [
                record
                for label in (0, 1)
                for record in [item for item in self.index if item["label"] == label][
                    :limit_per_class
                ]
            ]
        if limit is not None:
            self.index = self.index[:limit]

    def _read_protocol(self):
        if not self.protocol_path.is_file():
            raise FileNotFoundError(f"Protocol not found: {self.protocol_path}")

        records = []
        with self.protocol_path.open(encoding="utf-8") as protocol:
            for line_number, line in enumerate(protocol, start=1):
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(
                        f"Malformed protocol line {line_number}: expected 5 fields"
                    )
                speaker_id, utterance_id, _, attack_id, label = fields
                if label not in {"bonafide", "spoof"}:
                    raise ValueError(f"Unknown label {label!r} on line {line_number}")
                records.append(
                    {
                        "speaker_id": speaker_id,
                        "utterance_id": utterance_id,
                        "attack_id": attack_id,
                        "label": int(label == "bonafide"),
                        "path": self.data_dir
                        / f"ASVspoof2019_LA_{self.split}"
                        / "flac"
                        / f"{utterance_id}.flac",
                    }
                )
        return records

    def __len__(self):
        return len(self.index)

    def _fix_waveform_length(self, waveform):
        target_length = self.n_fft + (self.n_frames - 1) * self.hop_length
        length = waveform.shape[-1]
        if length < target_length:
            return torch.nn.functional.pad(waveform, (0, target_length - length))
        if length == target_length:
            return waveform
        max_start = length - target_length
        start = (
            torch.randint(max_start + 1, size=()).item()
            if self.random_crop
            else max_start // 2
        )
        return waveform[..., start : start + target_length]

    def __getitem__(self, index):
        record = self.index[index]
        waveform, sample_rate = torchaudio.load(record["path"])
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"{record['path']} has sample rate {sample_rate}, "
                f"expected {self.sample_rate}"
            )
        waveform = waveform.mean(dim=0)
        waveform = self._fix_waveform_length(waveform)
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        log_power = torch.log(spectrum.abs().square().clamp_min(1e-12))
        return {
            "data_object": log_power.unsqueeze(0),
            "labels": record["label"],
            "utterance_id": record["utterance_id"],
        }
