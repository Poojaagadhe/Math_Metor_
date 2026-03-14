# Math Mentor - AI-Powered Math Problem Solver

A reliable multimodal AI application that solves JEE-style math problems with step-by-step explanations, featuring RAG, multi-agent system, human-in-the-loop, and memory capabilities.

## Features

- 🖼️ **Image Input**: Upload photos or screenshots of math problems with OCR
- 🎤 **Audio Input**: Speak your math questions with automatic transcription
- ⌨️ **Text Input**: Type problems directly
- 🤖 **Multi-Agent System**: 5 specialized agents (Parser, Router, Solver, Verifier, Explainer)
- 📚 **RAG Pipeline**: Retrieves relevant formulas and solution patterns
- 🔄 **Human-in-the-Loop**: Intervention when confidence is low
- 🧠 **Memory & Learning**: Learns from past solutions and feedback
- ✅ **Solution Verification**: Validates answers before presenting

## Supported Topics

- Algebra (equations, polynomials, inequalities)
- Calculus (limits, derivatives, optimization)
- Probability (basic probability, combinations, permutations)
- Linear Algebra (matrices, vectors, transformations)

## Installation

### Prerequisites

- Python 3.9 or higher
- LLM_PROVIDER = _get("LLM_PROVIDER", "groq")


### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd math-mentor
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
# Required: OPENAI_API_KEY
```

5. **Initialize the knowledge base**
```bash
python -m rag.vector_store --init
```

## Running Locally

```bash
streamlit run ui/app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage Guide

### 1. Image Input
- Click the "Image" tab
- Upload a JPG/PNG image of a math problem
- Review the extracted text (edit if needed)
- Click "Solve" to get the solution

### 2. Audio Input
- Click the "Audio" tab
- Upload an audio file or record directly
- Review the transcription (edit if needed)
- Click "Solve" to get the solution

### 3. Text Input
- Click the "Text" tab
- Type your math problem
- Click "Solve" to get the solution

### 4. Understanding Results
- **Agent Trace**: See which agents ran and their outputs
- **Retrieved Context**: View relevant formulas/patterns from knowledge base
- **Solution**: Final answer with confidence score
- **Explanation**: Step-by-step breakdown
- **Feedback**: Mark as correct/incorrect to help the system learn

## Project Structure

```
math-mentor/
├── agents/              # Multi-agent system
├── input_processors/    # Image, audio, text processing
├── rag/                # RAG pipeline and vector store
├── memory/             # Memory and learning system
├── knowledge_base/     # Curated math documents
├── ui/                 # Streamlit application
├── utils/              # Helper utilities
├── tests/              # Test suite
└── data/               # Runtime data (uploads, memory, vectors)
```

## Architecture

See [architecture.md](architecture.md) for detailed system design.


## Knowledge Base Customization

Add your own math reference documents:

1. Create markdown files in `knowledge_base/<topic>/`
2. Run re-indexing:
```bash
python -m rag.vector_store --reindex
```

## Deployment
 - app link -- https://mathmetor-bsdkks9stvkysklstavylg.streamlit.app/
## Troubleshooting

**OCR not working**
- Ensure image is clear and well-lit
- Try editing the extracted text manually


**Audio transcription issues**
- Use clear audio without background noise
- Speak math terms clearly ("x squared", not "x2")
- Edit transcript before solving

**Low confidence warnings**
- Review the extracted/transcribed text
- Provide additional context if needed
- Use HITL to correct and teach the system


## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request


