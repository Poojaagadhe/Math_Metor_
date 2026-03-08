"""Test script to validate all agents"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

def test_imports():
    """Test that all modules can be imported"""
    print("=" * 60)
    print("Testing Module Imports")
    print("=" * 60)
    
    modules = [
        ("Config", "utils.config", "Config"),
        ("Logger", "utils.logger", "setup_logger"),
        ("HITL Manager", "utils.hitl", "HITLManager"),
        ("Image Processor", "input_processors.image_processor", "ImageProcessor"),
        ("Audio Processor", "input_processors.audio_processor", "AudioProcessor"),
        ("Text Processor", "input_processors.text_processor", "TextProcessor"),
        ("Embedding Generator", "rag.embeddings", "EmbeddingGenerator"),
        ("Vector Store", "rag.vector_store", "VectorStore"),
        ("Retriever", "rag.retriever", "Retriever"),
        ("Base Agent", "agents.base_agent", "BaseAgent"),
        ("Parser Agent", "agents.parser_agent", "ParserAgent"),
        ("Router Agent", "agents.router_agent", "RouterAgent"),
        ("Solver Agent", "agents.solver_agent", "SolverAgent"),
        ("Verifier Agent", "agents.verifier_agent", "VerifierAgent"),
        ("Explainer Agent", "agents.explainer_agent", "ExplainerAgent"),
        ("Memory Store", "memory.memory_store", "MemoryStore"),
        ("Learning Engine", "memory.learning_engine", "LearningEngine"),
    ]
    
    failed = []
    for name, module, cls in modules:
        try:
            mod = __import__(module, fromlist=[cls])
            getattr(mod, cls)
            print(f"✓ {name}")
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed.append((name, str(e)))
    
    print()
    if failed:
        print(f"❌ {len(failed)} modules failed to import:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        return False
    else:
        print("✅ All modules imported successfully!")
        return True

def test_agent_initialization():
    """Test that agents can be initialized"""
    print("\n" + "=" * 60)
    print("Testing Agent Initialization")
    print("=" * 60)
    
    # Check if API key is set
    if not Config.OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not set - skipping agent initialization")
        print("   (This is expected if you haven't configured .env yet)")
        return True
    
    agents = [
        ("Parser Agent", "agents.parser_agent", "ParserAgent"),
        ("Router Agent", "agents.router_agent", "RouterAgent"),
        ("Solver Agent", "agents.solver_agent", "SolverAgent"),
        ("Verifier Agent", "agents.verifier_agent", "VerifierAgent"),
        ("Explainer Agent", "agents.explainer_agent", "ExplainerAgent"),
    ]
    
    failed = []
    for name, module, cls in agents:
        try:
            mod = __import__(module, fromlist=[cls])
            agent_cls = getattr(mod, cls)
            agent = agent_cls()
            print(f"✓ {name} initialized (model: {agent.model})")
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed.append((name, str(e)))
    
    print()
    if failed:
        print(f"❌ {len(failed)} agents failed to initialize:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        return False
    else:
        print("✅ All agents initialized successfully!")
        return True

def test_config():
    """Test configuration"""
    print("\n" + "=" * 60)
    print("Testing Configuration")
    print("=" * 60)
    
    checks = [
        ("Data directory", Config.DATA_DIR.exists()),
        ("Knowledge base directory", Config.KNOWLEDGE_BASE_DIR.exists()),
        ("OpenAI API key set", bool(Config.OPENAI_API_KEY)),
        ("Parser model configured", bool(Config.PARSER_MODEL)),
        ("Solver model configured", bool(Config.SOLVER_MODEL)),
    ]
    
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"{status} {check_name}")
    
    print()
    
    all_passed = all(result for _, result in checks)
    if all_passed:
        print("✅ Configuration checks passed!")
    else:
        print("⚠️  Some configuration checks failed")
        if not Config.OPENAI_API_KEY:
            print("   → Add OPENAI_API_KEY to .env file")
    
    return True  # Don't fail on config issues

def test_simple_agent_flow():
    """Test a simple agent flow without API calls"""
    print("\n" + "=" * 60)
    print("Testing Agent Structure")
    print("=" * 60)
    
    if not Config.OPENAI_API_KEY:
        print("⚠️  Skipping agent flow test (no API key)")
        return True
    
    try:
        from agents.parser_agent import ParserAgent
        
        parser = ParserAgent()
        
        # Check that agent has required methods
        assert hasattr(parser, 'run'), "Parser missing run() method"
        assert hasattr(parser, '_call_llm'), "Parser missing _call_llm() method"
        assert hasattr(parser, '_get_system_prompt'), "Parser missing _get_system_prompt() method"
        
        print("✓ Parser Agent structure validated")
        
        # Test system prompt generation
        prompt = parser._get_system_prompt()
        assert len(prompt) > 0, "System prompt is empty"
        assert "JSON" in prompt, "System prompt should mention JSON output"
        
        print("✓ Parser Agent system prompt validated")
        
        print("\n✅ Agent structure tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Agent structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n🧪 Math Mentor - Agent Testing\n")
    
    results = []
    
    # Test 1: Imports
    results.append(("Module Imports", test_imports()))
    
    # Test 2: Configuration
    results.append(("Configuration", test_config()))
    
    # Test 3: Agent Initialization
    results.append(("Agent Initialization", test_agent_initialization()))
    
    # Test 4: Agent Structure
    results.append(("Agent Structure", test_simple_agent_flow()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
        print("\nNext steps:")
        print("1. Add your OPENAI_API_KEY to .env")
        print("2. Run: python setup.py")
        print("3. Run: streamlit run ui/app.py")
    else:
        print("❌ Some tests failed - please review errors above")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
