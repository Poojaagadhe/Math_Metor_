from PIL import Image
import numpy as np

try:
    from pix2tex.cli import LatexOCR
    print("Pix2Tex imported successfully.")
except Exception as e:
    print(f"Failed to import Pix2Tex: {e}")
    exit(1)

try:
    model = LatexOCR()
    print("Pix2Tex model initialized.")
    # create a dummy image
    img = Image.new('RGB', (100, 30), color = (255, 255, 255))
    latex = model(img)
    print(f"Extraction successful: {latex}")
except Exception as e:
    print(f"Pix2Tex extraction failed: {e}")
    import traceback
    traceback.print_exc()
