# Rather than immediately processing 11,088 files, create a small test script.
from pathlib import Path

from src.data.inkml_parser import parse_inkml
from src.data.renderer import render_strokes


PROJECT_ROOT = Path(__file__).resolve().parents[1]


INKML_FILE = (
    PROJECT_ROOT
    / "data"
    / "crohme2019"
    / "train"
    / "65_herbert.inkml"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "images"
    / "65_herbert.png"
)


result = parse_inkml(INKML_FILE)

render_strokes(
    strokes=result["strokes"],
    output_path=OUTPUT_FILE,
)

print("Rendered image:")
print(OUTPUT_FILE)

print("LaTeX:")
print(result["latex"])