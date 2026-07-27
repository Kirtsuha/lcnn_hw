import pytest
import torch

from src.model.lcnn import LCNN, MaxFeatureMap


def test_mfm_selects_pairwise_maximum():
    layer = MaxFeatureMap()
    inputs = torch.tensor([[[1.0], [4.0], [3.0], [2.0]]])
    assert torch.equal(layer(inputs), torch.tensor([[[3.0], [4.0]]]))


def test_mfm_rejects_odd_feature_count():
    with pytest.raises(ValueError):
        MaxFeatureMap()(torch.zeros(2, 3))


def test_lcnn_output_shape():
    model = LCNN(input_frequency_bins=257, input_frames=750)
    model.eval()
    with torch.no_grad():
        outputs = model(torch.randn(2, 1, 257, 750))
    assert outputs["logits"].shape == (2, 2)
