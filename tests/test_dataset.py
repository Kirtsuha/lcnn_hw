from pathlib import Path

import pytest
import torch

from src.datasets.asvspoof import ASVspoofDataset
from src.datasets.data_utils import build_balanced_sampler


def test_protocol_parser(tmp_path: Path):
    protocol = tmp_path / "protocol.txt"
    protocol.write_text("LA_001 LA_T_0000001 - A01 bonafide\n", encoding="utf-8")
    dataset = ASVspoofDataset(tmp_path, protocol, split="train")
    assert len(dataset) == 1
    assert dataset.index[0]["label"] == 1
    assert dataset.index[0]["utterance_id"] == "LA_T_0000001"


def test_malformed_protocol_is_rejected(tmp_path: Path):
    protocol = tmp_path / "protocol.txt"
    protocol.write_text("not an official protocol\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ASVspoofDataset(tmp_path, protocol, split="train")


def test_class_balanced_limit(tmp_path: Path):
    protocol = tmp_path / "protocol.txt"
    protocol.write_text(
        "\n".join(
            [
                "S U1 - A01 bonafide",
                "S U2 - A01 bonafide",
                "S U3 - A01 spoof",
                "S U4 - A01 spoof",
            ]
        ),
        encoding="utf-8",
    )
    dataset = ASVspoofDataset(tmp_path, protocol, split="train", limit_per_class=1)
    assert sorted(item["label"] for item in dataset.index) == [0, 1]


def test_balanced_sampler_uses_inverse_class_frequencies(tmp_path: Path):
    protocol = tmp_path / "protocol.txt"
    protocol.write_text(
        "\n".join(
            [
                "S U1 - A01 bonafide",
                "S U2 - A01 spoof",
                "S U3 - A01 spoof",
                "S U4 - A01 spoof",
            ]
        ),
        encoding="utf-8",
    )
    dataset = ASVspoofDataset(tmp_path, protocol, split="train")

    sampler = build_balanced_sampler(dataset)

    assert sampler.num_samples == 4
    assert sampler.replacement
    assert torch.equal(
        sampler.weights,
        torch.tensor([1.0, 1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=torch.double),
    )


def test_balanced_sampler_rejects_single_class_dataset(tmp_path: Path):
    protocol = tmp_path / "protocol.txt"
    protocol.write_text(
        "\n".join(
            [
                "S U1 - A01 spoof",
                "S U2 - A01 spoof",
            ]
        ),
        encoding="utf-8",
    )
    dataset = ASVspoofDataset(tmp_path, protocol, split="train")

    with pytest.raises(ValueError, match="at least two classes"):
        build_balanced_sampler(dataset)
