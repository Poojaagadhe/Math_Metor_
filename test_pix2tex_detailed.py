import traceback
from PIL import Image
from pix2tex.cli import LatexOCR
import numpy as np
import cv2

try:
    model = LatexOCR()
    print("Pix2Tex model initialized.")
    # create a dummy image
    img = Image.new('RGB', (100, 30), color = (255, 255, 255))
    latex = model(img)
    print(f"Extraction successful: {latex}")
except Exception as e:
    print(f"Pix2Tex extraction failed: {e}")
    traceback.print_exc()
