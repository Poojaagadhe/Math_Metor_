from PIL import Image
import numpy as np

def _is_image_empty(image: Image.Image) -> bool:
    """Check if an image is completely uniform (e.g., pure white)."""
    # Convert to grayscale
    gray = image.convert('L')
    # Get extremum (min, max) values
    extrema = gray.getextrema()
    # If min == max, the image is a single solid color
    return extrema[0] == extrema[1]

# Test cases
img_white = Image.new('RGB', (100, 30), color=(255, 255, 255))
img_black = Image.new('RGB', (100, 30), color=(0, 0, 0))
img_text = Image.new('RGB', (100, 30), color=(255, 255, 255))
img_text.putpixel((50, 15), (0, 0, 0))

print(f"White image empty: {_is_image_empty(img_white)}")
print(f"Black image empty: {_is_image_empty(img_black)}")
print(f"Text image empty: {_is_image_empty(img_text)}")
