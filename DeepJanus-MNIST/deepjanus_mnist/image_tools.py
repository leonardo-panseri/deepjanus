from xml.etree import ElementTree as ET

import cairosvg
import numpy as np
import potrace
from PIL import Image
from io import BytesIO


def svg_path_from_potrace_path(potrace_path: potrace.Path) -> str:
    """Converts a Potrace path to an SVG path."""
    def coord_pair(coords):
        return f'{int(coords.x)},{int(coords.y)}'

    # Build a SVG path description:
    # M - Move To <point>
    # L - Line To <point>
    # C - Bézier Curve with <control_point_1> <control_point_2> To <end>
    svg_path = []
    curve: potrace.Curve
    for curve in potrace_path:
        start = curve.start_point
        svg_path.append(f' M{coord_pair(start)}')
        segment: potrace.CornerSegment | potrace.BezierSegment
        for segment in curve:
            if segment.is_corner:
                corner = segment.c
                end = segment.end_point
                svg_path.append(f' L{coord_pair(corner)}'
                                f' L{coord_pair(end)}')
            else:
                control_point_1 = segment.c1
                control_point_2 = segment.c2
                end = segment.end_point
                svg_path.append(f' C{coord_pair(control_point_1)}'
                                f' {coord_pair(control_point_2)}'
                                f' {coord_pair(end)}')
        svg_path.append(' Z')
        return ''.join(svg_path)


def create_svg_xml(svg_path: str) -> str:
    """Creates the XML representation of an SVG image containing a single path."""
    root = ET.Element('svg')
    root.set('version', '1.0')
    root.set("xmlns", "http://www.w3.org/2000/svg")
    root.set("height", str(28))
    root.set("width", str(28))
    path = ET.SubElement(root, "path")
    path.set("d", svg_path)
    xml_str = ET.tostring(root, encoding='unicode', method='xml')
    return xml_str


def bitmap_to_svg(bitmap: np.ndarray) -> str:
    """Converts a grayscale bitmap image to a SVG image."""
    potrace_bitmap = potrace.Bitmap(bitmap)
    potrace_bitmap.invert()
    potrace_path = potrace_bitmap.trace()
    svg_path = svg_path_from_potrace_path(potrace_path)
    return create_svg_xml(svg_path)


def svg_to_bitmap(svg: str) -> np.ndarray:
    """Converts a SVG image to a grayscale bitmap image."""
    # Rasterize SVG image into buffer
    buffer = BytesIO()
    cairosvg.svg2png(bytestring=svg, write_to=buffer)
    # Use Pillow to obtain a NumPy array from the rasterized image
    pillow_image = Image.open(buffer)
    rgba = np.asarray(pillow_image)
    # CairoSVG returns an image in RGBA format.
    # We can easily convert to grayscale by keeping only the A channel
    bitmap = rgba[:,:,3]
    return bitmap


def calculate_bitmap_distance(bitmap1: np.ndarray, bitmap2: np.ndarray):
    """Measures the 'distance' between two grayscale bitmaps (how different they are)."""
    return np.linalg.norm(bitmap1- bitmap2)


if __name__ == '__main__':
    import h5py

    d = h5py.File('../data/mnist/digit5.h5', 'r')
    x = np.array(d.get('xn'))
    Image.fromarray(x[0]).save('test_original.png')
    s = bitmap_to_svg(x[0])
    with open('test_vector.svg', 'w') as file:
        file.write(s)

    b = svg_to_bitmap(s)
    Image.fromarray(b).save('test_raster.png')
