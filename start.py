"""Quick start script for Math Mentor"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    """Quick start Math Mentor"""
    print("🚀 Starting Math Mentor...")
    
    # Check if .env exists
    env_file = Path(".env")
    if not env_file.exists():
        print("\n⚠️  .env file not found!")
        print("Creating .env from .env.example...")
        
        example_file = Path(".env.example")
        if example_file.exists():
            import shutil
            shutil.copy(example_file, env_file)
            print("✓ Created .env file")
            print("\n⚠️  Please edit .env and add your OPENAI_API_KEY")
            print("Then run this script again.")
            return
        else:
            print("❌ .env.example not found!")
            return
    
    # Check if knowledge base is initialized
    chroma_dir = Path("data/chroma_db")
    if not chroma_dir.exists() or not list(chroma_dir.iterdir()):
        print("\n📚 Knowledge base not initialized")
        print("Running setup...")
        subprocess.run([sys.executable, "setup.py"])
    
    # Start Streamlit
    print("\n🎯 Launching Streamlit app...")
    print("=" * 60)
    subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/app.py"])

if __name__ == "__main__":
    main()
