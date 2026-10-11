"""Unit tests for src/inference.py.

Run from the HMER project root:
    python -m pytest tests/test_inference.py -q
"""
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.inference import (
    BaselineHMER,
    HMERInference,
    decode_token_ids,
    load_vocabulary,
    preprocess_image,
)


def _write_vocab(path: Path):
    token_to_id = {
        "<PAD>": 0,
        "<SOS>": 1,
        "<EOS>": 2,
        "<UNK>": 3,
        "x": 4,
        "+": 5,
    }
    path.write_text(json.dumps({"token_to_id": token_to_id}), encoding="utf-8")
    return token_to_id


def test_load_vocabulary_supports_canonical_structure(tmp_path):
    vocab_path = tmp_path / "vocab.json"
    token_to_id = _write_vocab(vocab_path)
    loaded, id_to_token = load_vocabulary(vocab_path)
    assert loaded == token_to_id
    assert id_to_token[4] == "x"


def test_load_vocabulary_rejects_missing_special_token(tmp_path):
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps({"token_to_id": {"x": 0}}), encoding="utf-8")
    with pytest.raises(ValueError, match="special token"):
        load_vocabulary(vocab_path)


def test_preprocess_image_preserves_aspect_ratio_and_scales_height():
    image = Image.new("RGB", (100, 50), color="white")
    tensor = preprocess_image(image, image_height=128, max_image_width=1024)
    assert tuple(tensor.shape) == (1, 128, 256)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) == pytest.approx(1.0)
    assert float(tensor.max()) == pytest.approx(1.0)


def test_preprocess_image_rejects_excessive_width():
    image = Image.new("L", (2000, 100), color="white")
    with pytest.raises(ValueError, match="exceeds"):
        preprocess_image(image, image_height=128, max_image_width=1000)


def test_decode_token_ids_skips_special_tokens_and_stops_at_eos():
    mapping = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>", 4: "x", 5: "+"}
    result = decode_token_ids([1, 4, 5, 2, 4], mapping, 0, 1, 2)
    assert result == "x +"


def test_checkpoint_load_and_predict_smoke(tmp_path):
    vocab_path = tmp_path / "vocab.json"
    token_to_id = _write_vocab(vocab_path)
    model_dir = tmp_path / "outputs" / "models"
    model_dir.mkdir(parents=True)
    checkpoint_path = model_dir / "best_model.pt"

    model = BaselineHMER(
        vocab_size=len(token_to_id),
        embedding_dim=16,
        hidden_dim=32,
        num_layers=1,
        dropout=0.0,
        pad_id=token_to_id["<PAD>"],
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": len(token_to_id),
                "embedding_dim": 16,
                "hidden_dim": 32,
                "num_layers": 1,
                "dropout": 0.0,
            },
        },
        checkpoint_path,
    )

    predictor = HMERInference(
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        project_root=tmp_path,
        device="cpu",
    )
    result = predictor.predict(Image.new("L", (120, 50), color="white"), max_length=3)
    assert isinstance(result["latex"], str)
    assert isinstance(result["token_ids"], list)
    assert isinstance(result["terminated"], bool)
    assert result["checkpoint"] == str(checkpoint_path.resolve())
