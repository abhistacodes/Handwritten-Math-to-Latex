from pathlib import Path
import xml.etree.ElementTree as ET


INKML_NAMESPACE = {
    "ink": "http://www.w3.org/2003/InkML"
}


def parse_trace(trace_text):
    """
    Parse the coordinate string from an InkML <trace> element.

    Example:
        "337 98, 335 98, 340 96"

    Returns:
        list[tuple[float, float]]
    """

    points = []

    if not trace_text:
        return points

    for point in trace_text.strip().split(","):
        values = point.strip().split()

        if len(values) < 2:
            continue

        x = float(values[0])
        y = float(values[1])

        points.append((x, y))

    return points


def parse_inkml(file_path):
    """
    Parse a CROHME InkML file.

    Returns:
        dict containing:
            - file_name
            - expression
            - latex
            - strokes
    """

    file_path = Path(file_path)

    tree = ET.parse(file_path)
    root = tree.getroot()

    
    # ---------------------------------------------------------
    # Extract LaTeX ground truth
    # ---------------------------------------------------------

    latex = None

    for annotation in root.findall("ink:annotation", INKML_NAMESPACE):

        annotation_type = annotation.attrib.get("type")

        if annotation_type == "truth":
            latex = annotation.text.strip() if annotation.text else ""

            break

    
    # ---------------------------------------------------------
    # Extract expression ID
    # ---------------------------------------------------------

    expression = None

    for annotation in root.findall("ink:annotation", INKML_NAMESPACE):

        annotation_type = annotation.attrib.get("type")

        if annotation_type == "expression":
            expression = annotation.text.strip() if annotation.text else ""

            break

    
    # ---------------------------------------------------------
    # Extract strokes
    # ---------------------------------------------------------

    strokes = []

    for trace in root.findall("ink:trace", INKML_NAMESPACE):

        points = parse_trace(trace.text)

        if points:
            strokes.append(points)

    return {
        "file_name": file_path.name,
        "expression": expression,
        "latex": latex,
        "strokes": strokes,
    }


if __name__ == "__main__":

    sample_file = (
        Path("data")
        / "crohme2019"
        / "train"
        / "65_herbert.inkml"
    )

    result = parse_inkml(sample_file)

    print("File:", result["file_name"])
    print("Expression:", result["expression"])
    print("LaTeX:", result["latex"])
    print("Number of strokes:", len(result["strokes"]))

    for i, stroke in enumerate(result["strokes"][:3]):
        print(f"Stroke {i}: {len(stroke)} points")