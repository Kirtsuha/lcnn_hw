from pathlib import Path

import pytest

from src.datasets.asvspoof import ASVspoofDataset


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
