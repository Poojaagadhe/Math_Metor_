
import time
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def measure_startup():
    print("🚀 Measuring startup time (import speed)...")
    start_time = time.time()
    
    # Simulate the main imports in app.py after optimization
    import streamlit as st
    from input_processors.text_processor import TextProcessor
    from agents.parser_agent import ParserAgent
    from agents.router_agent import RouterAgent
    from agents.solver_agent import SolverAgent
    from agents.verifier_agent import VerifierAgent
    from agents.explainer_agent import ExplainerAgent
    from memory.memory_store import MemoryStore
    from memory.learning_engine import LearningEngine
    from utils.hitl import HITLManager
    
    end_time = time.time()
    print(f"✅ Core imports completed in {end_time - start_time:.4f} seconds.")
    
    print("\n📦 Checking if heavy models are loaded...")
    # These should NOT be in sys.modules if lazy loading works
    heavy_modules = ['easyocr', 'pix2tex']
    for mod in heavy_modules:
        if mod in sys.modules:
            print(f"⚠️  {mod} is ALREADY loaded (not lazy)! Check your imports.")
        else:
            print(f"✅ {mod} is NOT loaded yet.")

    print("\n🛠️  Simulating on-demand loading of ImageProcessor...")
    from input_processors.image_processor import ImageProcessor
    ip_init_start = time.time()
    ip = ImageProcessor()
    ip_init_end = time.time()
    print(f"✅ ImageProcessor initialized in {ip_init_end - ip_init_start:.4f} seconds (should be fast).")
    
    print("\n🔍 Simulating first OCR call (this should trigger heavy load)...")
    ocr_start = time.time()
    # Accessing the reader property should trigger the lazy load
    _ = ip.reader
    ocr_end = time.time()
    print(f"✅ EasyOCR loaded on-demand in {ocr_end - ocr_start:.4f} seconds.")

if __name__ == "__main__":
    measure_startup()
