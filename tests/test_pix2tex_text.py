import traceback
from PIL import Image, ImageDraw
from pix2tex.cli import LatexOCR

try:
    model = LatexOCR()

    # create an image with some text/drawing so it isn't cropped to size 0
    img = Image.new('RGB', (200, 100), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((50, 40), "f(x) = x^2", fill=(0, 0, 0))

    print("Image created. Running OCR...")
    latex = model(img)
    print(f"Extraction successful: {latex}")
except Exception as e:
    print(f"Pix2Tex extraction failed: {e}")
    traceback.print_exc()
