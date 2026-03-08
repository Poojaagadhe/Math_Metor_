# Math Mentor - Quick Reference

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
# Edit .env and add: OPENAI_API_KEY=your_key_here

# 3. Initialize knowledge base
python setup.py

# 4. Run application
python start.py
# OR
streamlit run ui/app.py
```

## 📁 Project Structure

```
math-mentor/
├── agents/              # 5 AI agents (Parser, Router, Solver, Verifier, Explainer)
├── input_processors/    # Image OCR, Audio ASR, Text processing
├── rag/                # Vector store (ChromaDB) + retrieval
├── memory/             # SQLite storage + learning engine
├── knowledge_base/     # Math reference documents (5 topics)
├── ui/                 # Streamlit app + components
├── utils/              # Config, logging, HITL
└── data/               # Runtime data (uploads, DB, vectors)
```

## 🔑 Key Files

- **`ui/app.py`**: Main Streamlit application
- **`setup.py`**: Initialize knowledge base
- **`start.py`**: Quick launch script
- **`.env`**: Configuration (add your API key here!)
- **`requirements.txt`**: All dependencies

## 🎯 Main Features

### Input Modes
1. **Text**: Type directly
2. **Image**: Upload photo → OCR extraction
3. **Audio**: Upload/record → Speech-to-text

### Agent Pipeline
1. **Parser**: Structures problem
2. **Router**: Classifies & routes
3. **Solver**: Generates solution (uses RAG)
4. **Verifier**: Validates correctness
5. **Explainer**: Creates explanation

### Special Features
- **HITL**: Human review when confidence low
- **Memory**: Stores all solved problems
- **Learning**: Reuses patterns from past solutions
- **RAG**: Retrieves relevant formulas from knowledge base

## 🛠️ Common Commands

```bash
# Check dependencies
python setup.py --check-only

# Reinitialize knowledge base
python setup.py --force-reindex

# Run app
streamlit run ui/app.py

# View vector store stats
python -m rag.vector_store --stats
```

## ⚙️ Configuration (.env)

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
OCR_CONFIDENCE_THRESHOLD=0.7
VERIFIER_CONFIDENCE_THRESHOLD=0.8
WHISPER_MODEL=base
DEBUG=False
```

## 📊 Supported Topics

- **Algebra**: Equations, polynomials, inequalities
- **Calculus**: Limits, derivatives, optimization
- **Probability**: Basic probability, combinations, Bayes
- **Linear Algebra**: Matrices, vectors, transformations

## 🔍 Troubleshooting

**"OPENAI_API_KEY not found"**
→ Edit `.env` and add your API key

**"Failed to initialize components"**
→ Run `python setup.py --check-only`

**"No documents in knowledge base"**
→ Run `python setup.py` to load knowledge base

**OCR not working**
→ Ensure image is clear, well-lit, and high resolution

**Audio transcription issues**
→ Use clear audio without background noise

## 📦 Deployment (Streamlit Cloud)

1. Push to GitHub
2. Go to https://share.streamlit.io/
3. Connect repository
4. Add secrets:
   - `OPENAI_API_KEY = "your_key"`
5. Deploy!

## 📝 Next Steps

1. ✅ Add your OpenAI API key to `.env`
2. ✅ Run `python setup.py` to initialize
3. ✅ Test with `python start.py`
4. ⏳ Deploy to Streamlit Cloud
5. ⏳ Record demo video
6. ⏳ Create evaluation summary

## 🎥 Demo Video Outline

1. **Intro** (30s): Show UI, explain features
2. **Text Input** (45s): Type problem → Solution
3. **Image Input** (60s): Upload → OCR → Edit → Solve
4. **Audio Input** (60s): Record → Transcribe → Solve
5. **HITL** (45s): Trigger low confidence → Review
6. **Memory** (45s): Show similar problems reused
7. **Conclusion** (30s): Summary

Total: ~5 minutes

## 📚 Documentation

- **README.md**: Full installation & usage guide
- **architecture.md**: System design & diagrams
- **walkthrough.md**: Implementation details
- **implementation_plan.md**: Original design plan

## 💡 Tips

- Start with text input to test basic functionality
- Use clear, well-formatted images for OCR
- Speak math terms clearly for audio ("x squared" not "x2")
- Review extracted text before solving
- Provide feedback to help the system learn
- Check similar problems sidebar for patterns

## 🐛 Known Limitations

- Requires OpenAI API key (costs ~$0.05-0.15 per problem)
- OCR accuracy depends on image quality
- No multi-user authentication
- Local storage only (not production-ready for scale)

## 🚀 Future Enhancements

- LaTeX output support
- Diagram understanding
- Web search integration
- Teacher dashboard
- Model fine-tuning on feedback
- Multi-language support
