"""Setup script for Math Mentor"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import Config
from rag.vector_store import VectorStore
import argparse

def setup_knowledge_base():
    """Initialize vector store with knowledge base"""
    print("🔧 Initializing Math Mentor...")
    print(f"📁 Data directory: {Config.DATA_DIR}")
    
    # Ensure directories exist
    Config.ensure_directories()
    print("✓ Directories created")
    
    # Initialize vector store
    print("\n📚 Loading knowledge base into vector store...")
    store = VectorStore()
    
    # Load knowledge base
    store.load_knowledge_base()
    
    # Get stats
    stats = store.get_collection_stats()
    print(f"\n✓ Knowledge base loaded successfully!")
    print(f"  - Collection: {stats['collection_name']}")
    print(f"  - Documents: {stats['document_count']}")
    print(f"  - Location: {stats['persist_directory']}")
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Copy .env.example to .env")
    print("2. Add your OPENAI_API_KEY to .env")
    print("3. Run: streamlit run ui/app.py")

def check_dependencies():
    """Check if all dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required = [
        'streamlit',
        'openai',
        'chromadb',
        'easyocr',
        'PIL',
        'numpy',
        'sympy',
        'langchain',
        'sentence_transformers'
    ]
    
    missing = []
    for package in required:
        try:
            if package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed")
        return True

def check_env():
    """Check environment configuration"""
    print("\n🔍 Checking environment configuration...")
    
    if not Config.OPENAI_API_KEY:
        print("  ⚠️  OPENAI_API_KEY not set")
        print("  Please add your API key to .env file")
        return False
    else:
        print("  ✓ OPENAI_API_KEY configured")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup Math Mentor")
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies and config")
    parser.add_argument("--force-reindex", action="store_true", help="Force reindex knowledge base")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Math Mentor Setup")
    print("=" * 60)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n❌ Please install missing dependencies first")
        sys.exit(1)
    
    # Check environment
    env_ok = check_env()
    
    if args.check_only:
        if deps_ok and env_ok:
            print("\n✅ All checks passed!")
        else:
            print("\n⚠️  Some checks failed")
        sys.exit(0)
    
    if not env_ok:
        print("\n⚠️  Environment not fully configured, but continuing with setup...")
    
    # Setup knowledge base
    try:
        if args.force_reindex:
            print("\n🔄 Force reindexing knowledge base...")
            store = VectorStore()
            store.clear_collection()
        
        setup_knowledge_base()
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
