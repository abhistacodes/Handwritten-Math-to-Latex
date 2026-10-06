# Rather than immediately processing 11,088 files, create a small test script

from pathlib import Path
from PIL import Image, ImageDraw


def render_strokes(
    strokes,
    output_path,
    image_size=(128, 512),
    padding=20,
    line_width=2,
):
    """
    Render InkML strokes as a grayscale PNG image.

    Parameters
    ----------
    strokes : list[list[tuple[float, float]]]
        List of strokes, where each stroke contains (x, y) points.

    output_path : str or Path
        Path where the rendered PNG will be saved.

    image_size : tuple[int, int]
        Output image size as (height, width).

    padding : int
        Padding around the handwritten expression.

    line_width : int
        Width of the rendered strokes.
    """

    if not strokes:
        raise ValueError("No strokes provided.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Collect all points
    # ---------------------------------------------------------

    all_points = [
        point
        for stroke in strokes
        for point in stroke
    ]

    if not all_points:
        raise ValueError("No points found in strokes.")

    # ---------------------------------------------------------
    # Find bounding box
    # ---------------------------------------------------------

    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    # ---------------------------------------------------------
    # Normalize coordinates
    # ---------------------------------------------------------

    height, width = image_size

    available_width = width - 2 * padding
    available_height = height - 2 * padding

    drawing_width = max_x - min_x
    drawing_height = max_y - min_y

    # Avoid division by zero
    drawing_width = max(drawing_width, 1)
    drawing_height = max(drawing_height, 1)

    scale_x = available_width / drawing_width
    scale_y = available_height / drawing_height

    # Preserve aspect ratio
    scale = min(scale_x, scale_y)

    # ---------------------------------------------------------
    # Calculate centered position
    # ---------------------------------------------------------

    scaled_width = drawing_width * scale
    scaled_height = drawing_height * scale

    offset_x = (width - scaled_width) / 2
    offset_y = (height - scaled_height) / 2

    # ---------------------------------------------------------
    # Create grayscale canvas
    # ---------------------------------------------------------

    image = Image.new(
        "L",
        (width, height),
        color=255,
    )

    draw = ImageDraw.Draw(image)

    # ---------------------------------------------------------
    # Draw each stroke
    # ---------------------------------------------------------

    for stroke in strokes:

        if len(stroke) < 2:
            continue

        transformed_points = []

        for x, y in stroke:

            new_x = (x - min_x) * scale + offset_x
            new_y = (y - min_y) * scale + offset_y

            transformed_points.append(
                (round(new_x), round(new_y))
            )

        draw.line(
            transformed_points,
            fill=0,
            width=line_width,
            joint="curve",
        )

    # ---------------------------------------------------------
    # Save image
    # ---------------------------------------------------------

    image.save(output_path)

    return image