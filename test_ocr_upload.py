import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from input_processors.math_ocr_processor import MathOCRProcessor

def test_ocr():
    processor = MathOCRProcessor()
    
    # We saw three jpgs of the exact same size in data/uploads
    img_path = Path("data/uploads/c2e00281-668d-4277-95d7-9d5319e14487.jpg")
    
    print(f"Running OCR on {img_path}...")
    result = processor.process_image(img_path)
    print("Result:", result)

if __name__ == "__main__":
    test_ocr()
