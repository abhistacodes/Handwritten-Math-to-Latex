"""Inference utilities for the baseline CROHME HMER model.

Expected project layout:
    data/processed/crohme2019/metadata/vocab.json
    outputs/models/<checkpoint>.pt

The model architecture and image preprocessing mirror 04_baseline_model.ipynb.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


PathLike = Union[str, Path]


def find_project_root(start: Optional[PathLike] = None) -> Path:
    """Find the HMER project root by locating the expected data/src folders."""
    current = Path(start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "src").is_dir() and (
            (candidate / "data").exists() or (candidate / "outputs").exists()
        ):
            return candidate
    # Useful default when this module is imported from <project>/src.
    module_root = Path(__file__).resolve().parents[1]
    return module_root


def load_vocabulary(vocab_path: PathLike) -> tuple[Dict[str, int], Dict[int, str]]:
    """Load the canonical vocab.json structures used by preprocessing."""
    path = Path(vocab_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "token_to_id" in data:
        token_to_id = {str(k): int(v) for k, v in data["token_to_id"].items()}
    elif isinstance(data.get("vocabulary"), dict):
        token_to_id = {str(k): int(v) for k, v in data["vocabulary"].items()}
    else:
        token_to_id = {str(k): int(v) for k, v in data.items()}

    if "id_to_token" in data:
        id_to_token = {int(k): str(v) for k, v in data["id_to_token"].items()}
    else:
        id_to_token = {idx: token for token, idx in token_to_id.items()}

    for token in ("<PAD>", "<SOS>", "<EOS>", "<UNK>"):
        if token not in token_to_id:
            raise ValueError(f"Required special token {token!r} is missing from {path}")
    if len(set(token_to_id.values())) != len(token_to_id):
        raise ValueError(f"Vocabulary contains duplicate token IDs: {path}")
    return token_to_id, id_to_token


class CNNEncoder(nn.Module):
    """CNN encoder copied from the baseline model notebook."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(512, hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(images).flatten(1))


class LSTMDecoder(nn.Module):
    """LSTM decoder copied from the baseline model notebook."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.0,
        pad_id: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.image_to_hidden = nn.Linear(hidden_dim, hidden_dim * num_layers)
        self.image_to_cell = nn.Linear(hidden_dim, hidden_dim * num_layers)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def initialize_state(self, image_features: torch.Tensor):
        batch_size = image_features.size(0)
        hidden = self.image_to_hidden(image_features).view(
            self.num_layers, batch_size, self.hidden_dim
        )
        cell = self.image_to_cell(image_features).view(
            self.num_layers, batch_size, self.hidden_dim
        )
        return hidden, cell

    def forward(
        self, image_features: torch.Tensor, target_sequences: torch.Tensor
    ) -> torch.Tensor:
        embedded = self.embedding(target_sequences)
        hidden, cell = self.initialize_state(image_features)
        outputs, _ = self.lstm(embedded, (hidden, cell))
        return self.output_projection(outputs)


class BaselineHMER(nn.Module):
    """CNN encoder + LSTM decoder from 04_baseline_model.ipynb."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.0,
        pad_id: int = 0,
    ):
        super().__init__()
        self.encoder = CNNEncoder(hidden_dim=hidden_dim)
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_id=pad_id,
        )

    def forward(self, images: torch.Tensor, target_sequences: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(images), target_sequences)


def preprocess_image(
    image: Union[PathLike, Image.Image],
    image_height: int = 128,
    max_image_width: int = 1024,
) -> torch.Tensor:
    """Apply the same grayscale, aspect-ratio-preserving preprocessing as HMERDataset."""
    if image_height <= 0 or max_image_width <= 0:
        raise ValueError("image_height and max_image_width must be positive")

    if isinstance(image, Image.Image):
        pil_image = image.copy()
    else:
        image_path = Path(image).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        try:
            pil_image = Image.open(image_path)
            pil_image.load()
        except Exception as exc:
            raise ValueError(f"Could not read input image: {image_path}") from exc

    with pil_image:
        pil_image = pil_image.convert("L")
        width, height = pil_image.size
        if width < 1 or height < 1:
            raise ValueError("Input image has invalid dimensions")
        if height != image_height:
            scale = image_height / height
            new_width = max(1, round(width * scale))
            pil_image = pil_image.resize(
                (new_width, image_height), Image.Resampling.BILINEAR
            )
        if pil_image.width > max_image_width:
            raise ValueError(
                f"Resized image width {pil_image.width} exceeds "
                f"max_image_width={max_image_width}"
            )
        array = np.asarray(pil_image, dtype=np.float32) / 255.0

    # Model input: [channels=1, height, width]. Batch dimension is added at inference.
    return torch.from_numpy(array.copy()).unsqueeze(0)


def decode_token_ids(
    token_ids: Iterable[int],
    id_to_token: Dict[int, str],
    pad_id: int,
    sos_id: int,
    eos_id: int,
    unk_token: str = "<UNK>",
) -> str:
    """Decode IDs into the space-separated token format used by this project."""
    tokens: List[str] = []
    for raw_id in token_ids:
        idx = int(raw_id)
        if idx == eos_id:
            break
        if idx in (pad_id, sos_id):
            continue
        tokens.append(id_to_token.get(idx, unk_token))
    return " ".join(tokens)


def _extract_state_dict(checkpoint: Any) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Accept common PyTorch checkpoint payload formats."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a state dict or a checkpoint dictionary.")

    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        config = {}

    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value and all(
            isinstance(v, torch.Tensor) for v in value.values()
        ):
            return value, config

    # A raw state_dict is itself a dictionary of tensors.
    if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint, config

    raise ValueError(
        "Could not find model weights in checkpoint. Expected a raw state_dict "
        "or a dictionary containing 'model_state_dict', 'state_dict', or 'model'."
    )


def resolve_checkpoint_path(checkpoint_path: Optional[PathLike], project_root: Path) -> Path:
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_absolute():
            path = project_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    model_dir = project_root / "outputs" / "models"
    preferred = [
        model_dir / "best_model.pt",
        model_dir / "best_model.pth",
        model_dir / "best.pt",
        model_dir / "best.pth",
        model_dir / "checkpoint.pt",
        model_dir / "checkpoint.pth",
    ]
    for path in preferred:
        if path.is_file():
            return path
    candidates = sorted(
        [*model_dir.glob("*.pt"), *model_dir.glob("*.pth")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if model_dir.is_dir() else []
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No .pt/.pth checkpoint found in {model_dir}. Pass checkpoint_path explicitly."
    )


class HMERInference:
    """Load a trained baseline checkpoint and predict LaTeX from an expression image."""

    def __init__(
        self,
        checkpoint_path: Optional[PathLike] = None,
        vocab_path: Optional[PathLike] = None,
        project_root: Optional[PathLike] = None,
        device: Optional[str] = None,
        image_height: int = 128,
        max_image_width: int = 1024,
    ):
        self.project_root = Path(project_root).expanduser().resolve() if project_root else find_project_root()
        if vocab_path is None:
            vocab_path = self.project_root / "data" / "processed" / "crohme2019" / "metadata" / "vocab.json"
        else:
            vocab_path = Path(vocab_path).expanduser()
            if not vocab_path.is_absolute():
                vocab_path = self.project_root / vocab_path

        self.token_to_id, self.id_to_token = load_vocabulary(vocab_path)
        self.pad_id = self.token_to_id["<PAD>"]
        self.sos_id = self.token_to_id["<SOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        self.unk_id = self.token_to_id["<UNK>"]
        self.image_height = image_height
        self.max_image_width = max_image_width

        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = torch.device(device)

        checkpoint_file = resolve_checkpoint_path(checkpoint_path, self.project_root)
        try:
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        except TypeError:  # compatibility with older PyTorch versions
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
        state_dict, config = _extract_state_dict(checkpoint)

        # Defaults exactly match 04_baseline_model.ipynb; saved config overrides if present.
        embedding_dim = int(config.get("embedding_dim", 256))
        hidden_dim = int(config.get("hidden_dim", 512))
        num_layers = int(config.get("num_layers", 1))
        dropout = float(config.get("dropout", 0.0))
        checkpoint_vocab_size = int(config.get("vocab_size", len(self.token_to_id)))
        if checkpoint_vocab_size != len(self.token_to_id):
            raise ValueError(
                f"Checkpoint vocab_size={checkpoint_vocab_size} but vocabulary has "
                f"{len(self.token_to_id)} entries. Use the vocab.json used for training."
            )

        self.model = BaselineHMER(
            vocab_size=len(self.token_to_id),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_id=self.pad_id,
        ).to(self.device)
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint weights do not match the baseline architecture/configuration "
                "from 04_baseline_model.ipynb. Check the checkpoint's saved model config "
                "and ensure it was trained with this BaselineHMER implementation."
            ) from exc
        self.model.eval()
        self.checkpoint_path = checkpoint_file

    @torch.inference_mode()
    def predict(
        self,
        image: Union[PathLike, Image.Image],
        max_length: int = 200,
    ) -> Dict[str, Any]:
        """Predict one expression. Returns LaTeX, token IDs, and termination status."""
        if max_length < 1:
            raise ValueError("max_length must be at least 1")

        image_tensor = preprocess_image(
            image,
            image_height=self.image_height,
            max_image_width=self.max_image_width,
        ).unsqueeze(0).to(self.device)

        features = self.model.encoder(image_tensor)
        generated = [self.sos_id]
        terminated = False

        # This baseline decoder initializes its LSTM state from image features on each
        # forward call, so feed the full generated prefix and select its final timestep.
        for _ in range(max_length):
            prefix = torch.tensor([generated], dtype=torch.long, device=self.device)
            logits = self.model.decoder(features, prefix)
            next_id = int(torch.argmax(logits[0, -1], dim=-1).item())
            if next_id == self.eos_id:
                terminated = True
                break
            if next_id == self.sos_id or next_id == self.pad_id:
                # Prevent special input-only tokens from appearing in the expression.
                continue
            generated.append(next_id)

        token_ids = generated[1:]
        latex = decode_token_ids(
            token_ids,
            self.id_to_token,
            pad_id=self.pad_id,
            sos_id=self.sos_id,
            eos_id=self.eos_id,
        )
        return {
            "latex": latex,
            "token_ids": token_ids,
            "terminated": terminated,
            "checkpoint": str(self.checkpoint_path),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict LaTeX from a handwritten math image.")
    parser.add_argument("image", help="Path to a handwritten expression image.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path; defaults to outputs/models.")
    parser.add_argument("--vocab", default=None, help="Path to the training vocab.json.")
    parser.add_argument("--device", default=None, help="Device: cpu, cuda, or mps.")
    parser.add_argument("--max-length", type=int, default=200, help="Maximum generated tokens.")
    args = parser.parse_args()

    predictor = HMERInference(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        device=args.device,
    )
    result = predictor.predict(args.image, max_length=args.max_length)
    print("Predicted LaTeX:")
    print(result["latex"])
    print("EOS reached:", result["terminated"])


if __name__ == "__main__":
    main()
