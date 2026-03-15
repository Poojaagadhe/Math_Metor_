import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from input_processors.math_ocr_processor import MathOCRProcessor

print("Initializing MathOCRProcessor with local weights redirection...")
try:
    proc = MathOCRProcessor()
    model = proc.pix2tex_model
    if model:
        print("✅ Pix2Tex model loaded successfully with local weights!")
    else:
        print("❌ Model returned None.")
except Exception as e:
    print(f"❌ Failed to load model: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print(f"Checking if weights exist in project directory...")
weights_dir = Path("data/weights/pix2tex")
if weights_dir.exists():
    for f in weights_dir.iterdir():
        print(f" - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
else:
    print("❌ Weights directory not found!")
