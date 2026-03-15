import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from input_processors.math_ocr_processor import MathOCRProcessor
from PIL import Image
import traceback

print("Initializing MathOCRProcessor...")
try:
    proc = MathOCRProcessor()
    
    print("Getting Pix2Tex Model...")
    model = proc.pix2tex_model
    
    if model:
        print("Model loaded successfully!")
        
        # Test an actual image
        img = Image.new('RGB', (100, 30), color=(0, 0, 0))
        img.putpixel((50, 15), (255, 255, 255))
        
        try:
            res = model(img)
            print(f"Pix2Tex output: {res}")
        except Exception as e:
            print(f"Pix2Tex execution failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            
    else:
        print("Model returned None.")
except Exception as e:
    print(f"Failed to load model: {type(e).__name__}: {e}")
    traceback.print_exc()
