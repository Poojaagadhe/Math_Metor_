
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from input_processors.math_ocr_processor import MathOCRProcessor
from PIL import Image

def test_initialization():
    print("Testing MathOCRProcessor initialization...")
    try:
        processor = MathOCRProcessor()
        print("✅ Initialization successful!")
        return processor
    except NameError as e:
        print(f"❌ Initialization failed with NameError: {e}")
        return None
    except Exception as e:
        print(f"❌ Initialization failed with error: {e}")
        return None

def test_lazy_loading(processor):
    if not processor:
        return
    
    print("\nTesting lazy loading of EasyOCR...")
    try:
        reader = processor.easyocr_reader
        print("✅ EasyOCR reader loaded successfully!")
    except Exception as e:
        print(f"❌ EasyOCR loading failed: {e}")

    # Note: We won't test Pix2Tex directly if it's not installed or slow,
    # but the NameError was triggered by 'logger' which is now at the top.

if __name__ == "__main__":
    proc = test_initialization()
    test_lazy_loading(proc)
