from pathlib import Path
import random

from src.data.inkml_parser import parse_inkml
from src.data.renderer import render_strokes


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = (
    PROJECT_ROOT
    / "data"
    / "crohme2019"
    / "train"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "images"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Find all training InkML files
# ---------------------------------------------------------

inkml_files = list(TRAIN_DIR.glob("*.inkml"))

print(f"Training expressions found: {len(inkml_files)}")


# ---------------------------------------------------------
# Select 10 random expressions
# ---------------------------------------------------------

random.seed(42)

samples = random.sample(
    inkml_files,
    min(10, len(inkml_files)),
)


# ---------------------------------------------------------
# Render samples
# ---------------------------------------------------------

for index, inkml_file in enumerate(samples, start=1):

    result = parse_inkml(inkml_file)

    output_file = (
        OUTPUT_DIR
        / f"sample_{index:02d}.png"
    )

    render_strokes(
        strokes=result["strokes"],
        output_path=output_file,
    )

    print()
    print(f"Sample {index}")
    print(f"File:   {inkml_file.name}")
    print(f"LaTeX:  {result['latex']}")
    print(f"Strokes: {len(result['strokes'])}")
    print(f"Image:  {output_file}")