import traceback
from PIL import Image
from pix2tex.cli import LatexOCR
import numpy as np
import cv2

try:
    model = LatexOCR()
    img = Image.new('RGB', (100, 30), color = (255, 255, 255))
    latex = model(img)
    print("Success")
except Exception as e:
    with open("full_traceback.txt", "w") as f:
        traceback.print_exc(file=f)
    print("Traceback written to file.")
