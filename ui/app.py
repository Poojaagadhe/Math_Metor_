"""Main Streamlit application for Math Mentor"""

import streamlit as st
import sys
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Processors
from input_processors.image_processor import ImageProcessor
from input_processors.audio_processor import AudioProcessor
from input_processors.text_processor import TextProcessor
from input_processors.math_ocr_processor import MathOCRProcessor
from input_processors.vision_ocr_processor import VisionOCRProcessor

# Agents
from agents.parser_agent import ParserAgent
from agents.router_agent import RouterAgent
from agents.solver_agent import SolverAgent
from agents.verifier_agent import VerifierAgent
from agents.explainer_agent import ExplainerAgent
from agents.planner_agent import PlannerAgent

# Memory
from memory.memory_store import MemoryStore
from memory.learning_engine import LearningEngine

# Utilities
from utils.config import Config
from utils.hitl import HITLManager

# UI Components
from ui.components import (
    render_agent_trace,
    render_confidence_badge,
    render_retrieved_context,
    render_feedback_buttons,
    render_similar_problems
)

# -----------------------------
# Streamlit Page Setup
# -----------------------------

st.set_page_config(
    page_title="Math Mentor",
    page_icon="🧮",
    layout="wide"
)

st.title("🧮 Math Mentor")
st.caption("AI-Powered Math Problem Solver")

# -----------------------------
# Initialize Components (stored in session_state, not @st.cache_resource)
# session_state survives reruns within a session but resets on browser refresh
# or server restart — ensuring updated code is always picked up.

@st.cache_resource(show_spinner="Initializing Core Agents...")
def get_base_components():
    """Initialize lightweight agents and core utilities once per server session."""
    return {
        "text_processor": TextProcessor(),
        "parser_agent": ParserAgent(),
        "router_agent": RouterAgent(),
        "solver_agent": SolverAgent(),
        "verifier_agent": VerifierAgent(),
        "explainer_agent": ExplainerAgent(),
        "planner_agent": PlannerAgent(),
        "memory_store": MemoryStore(),
        "learning_engine": LearningEngine(),
        "hitl_manager": HITLManager(),
    }

def get_heavy_component(name):
    """Retrieve or initialize heavy ML components on demand."""
    if name not in st.session_state["heavy_components"]:
        with st.spinner(f"Loading {name.replace('_', ' ').title()}..."):
            if name == "image_processor":
                st.session_state["heavy_components"][name] = ImageProcessor()
            elif name == "audio_processor":
                st.session_state["heavy_components"][name] = AudioProcessor()
            elif name == "math_ocr":
                st.session_state["heavy_components"][name] = MathOCRProcessor()
            elif name == "vision_ocr":
                st.session_state["heavy_components"][name] = VisionOCRProcessor()
    
    return st.session_state["heavy_components"][name]

# Initialize base components if not present
if "base_components" not in st.session_state:
    st.session_state["base_components"] = get_base_components()

# Initialize container for heavy components if not present
if "heavy_components" not in st.session_state:
    st.session_state["heavy_components"] = {}

# Shortcut for compatibility with existing code
# We'll replace manual 'components' lookups for heavy items with a proxy or direct function call
# but keep 'base' items as is for minimal diff.
base_components = st.session_state["base_components"]

# -----------------------------
# Check Configuration Status
# -----------------------------

unconfigured_agents = [
    (name, agent) for name, agent in base_components.items() 
    if hasattr(agent, "is_configured") and not agent.is_configured
]

if unconfigured_agents:
    st.warning("⚠️ **System Configuration Required**")
    with st.expander("How to fix this error?", expanded=True):
        st.markdown("""
        Math Mentor needs an LLM API key to function. Since you are on Streamlit Cloud, you must add your keys to the dashboard secrets:
        
        1. Go to your **Streamlit Cloud Dashboard**.
        2. Open your app settings -> **Secrets**.
        3. Add your key like this:
        ```toml
        GROQ_API_KEY = "your_key_here"
        LLM_PROVIDER = "groq"
        ```
        4. Save and reboot the app.
        """)
        
        for name, agent in unconfigured_agents:
            st.error(f"**{name}** is not configured: {agent.config_error}")
            
    if st.button("🔄 Check Configuration / Refresh"):
        st.session_state.pop("base_components", None)
        st.session_state.pop("heavy_components", None)
        st.rerun()
    
    st.divider()
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("System Overview")

    st.markdown("""
    **Math Mentor uses:**
    - OCR for image problems
    - Speech transcription
    - Multi-Agent AI system
    - Retrieval Augmented Generation
    - Knowledge base reasoning
    - Human-in-the-loop verification
    """)

    stats = base_components["memory_store"].get_stats()

    st.metric("Problems Solved", stats.get("total_problems", 0))


# -----------------------------
# Core Processing Pipeline
# -----------------------------

def process_problem(problem_text, source, image_path=None, audio_path=None):

    st.divider()
    st.header("Processing Pipeline")

    current_text = problem_text
    max_retries = 2
    attempt = 0
    
    while attempt <= max_retries:
        if attempt > 0:
            st.warning(f"🔄 OCR Self-Correction Attempt {attempt}/{max_retries}...")
            
        trace = {}

        # ---------------- Parser Agent ----------------

        with st.status(f"Parser Agent Running... (Attempt {attempt+1})"):

            parsed = base_components["parser_agent"].run({
                "text": current_text,
                "source": source
            })

            trace["parser"] = {
                "status": "completed",
                "summary": f"Topic: {parsed.get('topic','unknown')}"
            }

        # ---------------- Router Agent ----------------

        with st.status(f"Router Agent Running... (Attempt {attempt+1})"):

            routing = base_components["router_agent"].run(parsed)

            trace["router"] = {
                "status": "completed",
                "summary": routing.get("task", routing.get("workflow", "default"))
            }

        # ---------------- Planner Agent ----------------

        with st.status(f"Planner Agent Running... (Attempt {attempt+1})"):
            
            plan = base_components["planner_agent"].run({
                "problem_text": parsed.get("problem_text"),
                "topic": parsed.get("topic"),
                "routing": routing
            })

            trace["planner"] = {
                "status": "completed",
                "summary": plan.get("strategy", "llm_reasoning")
            }

        # ---------------- Similar Problems ----------------

        similar = base_components["learning_engine"].find_similar_solved_problems(
            parsed.get("topic"),
            limit=3
        )

        if similar and attempt == 0:
            render_similar_problems(similar)

        # ---------------- Solver Agent ----------------

        with st.status(f"Solver Agent Running... (Attempt {attempt+1})"):

            solution = base_components["solver_agent"].run({
                "problem_text": parsed.get("problem_text"),
                "topic": parsed.get("topic"),
                "routing": routing,
                "parsed_data": parsed,
                "plan": plan
            })

            trace["solver"] = {
                "status": "completed",
                "summary": "Solution generated"
            }

        # ---------------- Verifier Agent ----------------

        with st.status(f"Verifier Agent Running... (Attempt {attempt+1})"):

            verification = base_components["verifier_agent"].run({
                "problem_text": parsed.get("problem_text"),
                "solution": solution.get("solution"),
                "steps": solution.get("steps"),
                "topic": parsed.get("topic")
            })

            trace["verifier"] = {
                "status": "completed",
                "summary": f"Confidence {verification.get('confidence',0):.2f}"
            }

        # ---------------- Reflection / Correction Loop ----------
        issues = verification.get("issues_found", [])
        is_correct = verification.get("is_correct", False)
        confidence = verification.get("confidence", 0.0)

        needs_correction = False
        feedback = []

        if not is_correct or confidence < 0.6:
            needs_correction = True
            feedback.extend(issues)
        if parsed.get("needs_clarification"):
            needs_correction = True
            feedback.append(parsed.get("clarification_needed", ""))
            
        if needs_correction and source == "image" and attempt < max_retries:
            st.info("⚠️ Verifier detected issues. Running automatic PlannerAgent OCR correction...")
            feedback_str = "\\n".join(filter(None, feedback))
            
            with st.status("Planner Agent correcting OCR..."):
                current_text = base_components["planner_agent"].plan_ocr_correction(
                    text=current_text,
                    feedback=feedback_str
                )
            
            attempt += 1
            continue
            
        break

    # ---------------- Explainer Agent ----------------

    with st.status("Explainer Agent Running..."):

        explanation = base_components["explainer_agent"].run({
            "problem_text": parsed.get("problem_text"),
            "solution": solution.get("solution"),
            "steps": solution.get("steps"),
            "topic": parsed.get("topic"),
            "retrieved_context": solution.get("retrieved_context",[])
        })

        trace["explainer"] = {
            "status": "completed",
            "summary": "Explanation generated"
        }

    # ---------------- Display Results ----------------

    st.divider()
    st.header("📊 Results")

    render_agent_trace(trace)

    if solution.get("retrieved_context"):
        render_retrieved_context(solution["retrieved_context"])

    # ── Confidence ──────────────────────────────────────────────────
    render_confidence_badge(verification.get("confidence", 0), "Solution Confidence")

    issues = verification.get("issues_found", [])
    if issues:
        with st.expander("⚠️ Verifier Notes"):
            for issue in issues:
                st.markdown(f"- {issue}")

    # ── Full Step-by-Step Solution ──────────────────────────────────
    st.subheader("🔢 Step-by-Step Solution")

    # Prefer the complete LLM response (has all steps) over the stripped final answer
    full = solution.get("full_solution") or solution.get("solution", "")
    if full:
        st.markdown(full)
    else:
        st.info("No solution could be generated. Please rephrase the problem.")

    # ── Student-Friendly Explanation ────────────────────────────────
    with st.expander("📖 Detailed Explanation", expanded=True):
        explanation_text = explanation.get("explanation", "")
        if explanation_text:
            st.markdown(explanation_text)
        else:
            st.info("No explanation available.")

        # Key concepts chip row
        key_concepts = explanation.get("key_concepts", [])
        if key_concepts:
            st.markdown("**Key Concepts:** " + " • ".join(f"`{c}`" for c in key_concepts))



    # ---------------- Memory Storage ----------------

    problem_id = str(uuid.uuid4())

    base_components["memory_store"].store_problem({
        "problem_id": problem_id,
        "input_type": source,
        # Align field names with memory_store.py column expectations
        "raw_input_path": image_path or audio_path,
        "extracted_text": problem_text,
        "topic": parsed.get("topic"),
        "subtopic": parsed.get("subtopic"),
        "solution": solution.get("solution"),
        "verifier_confidence": verification.get("confidence"),
        "hitl_triggered": verification.get("hitl_required", False)
    })

    render_feedback_buttons(problem_id)


# -----------------------------
# Tabs
# -----------------------------

tab1, tab2, tab3 = st.tabs(["Text", "Image", "Audio"])


# -----------------------------
# Text Tab
# -----------------------------

with tab1:

    problem = st.text_area(
        "Enter your math problem",
        height=150
    )

    if st.button("Solve Text Problem"):

        if problem.strip():
            process_problem(problem, "text")

        else:
            st.warning("⚠️ Please enter a math problem before solving.")


# -----------------------------
# Image Tab
# -----------------------------

with tab2:

    st.subheader("Upload Image of Math Problem")

    uploaded_image = st.file_uploader(
        "Upload math problem image",
        type=["png", "jpg", "jpeg"],
        key="image_uploader"
    )

    # Clear image session state when a new file is uploaded
    if uploaded_image:
        prev_name = st.session_state.get("image_last_filename")
        if prev_name != uploaded_image.name:
            st.session_state.pop("image_extracted_text", None)
            st.session_state.pop("image_ocr_method", None)
            st.session_state.pop("image_confidence", None)
            st.session_state.pop("image_save_path", None)
            st.session_state["image_last_filename"] = uploaded_image.name

    if uploaded_image:

        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        # -- Automated OCR caption --
        st.caption("The system automatically detects equations or text from the uploaded image.")

        # -- STEP 1: Extract button ---------------------------------
        if st.button("🔍 Extract Text from Image", key="extract_image_problem"):

            # Save uploaded file
            save_dir = Path("data/uploads")
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / uploaded_image.name

            with st.spinner("Saving image..."):
                with open(save_path, "wb") as f:
                    f.write(uploaded_image.getbuffer())

            extracted_text = ""
            ocr_method = "Unknown"
            confidence = 0.0

            # -- Automated Extraction via MathOCRProcessor --
            try:
                with st.spinner("Running automated OCR..."):
                    result = get_heavy_component("math_ocr").process_image(save_path)
                
                extracted_text = result.get("extracted_text", "")
                ocr_method = result.get("method", "Unknown").title()
                confidence = result.get("confidence", 0.0)
            except Exception as e:
                st.warning(f"⚠️ Automated OCR error: {e}")

            # Persist to session_state so it survives the next Streamlit rerun
            st.session_state["image_extracted_text"] = extracted_text
            st.session_state["image_ocr_method"] = ocr_method
            st.session_state["image_confidence"] = confidence
            st.session_state["image_save_path"] = str(save_path)

        # -- STEP 2: Show editor and Solve (OUTSIDE Extract block) --
        # These render on every run as long as session_state has results
        if st.session_state.get("image_extracted_text") is not None:

            st.success("✅ OCR Completed")
            st.write("**OCR Method:**", st.session_state["image_ocr_method"])
            st.write("**Confidence:**", round(st.session_state["image_confidence"], 3))

            st.subheader("Extracted Text / Equation")

            edited_text = st.text_area(
                "Review and edit if needed:",
                value=st.session_state["image_extracted_text"],
                height=150,
                key="image_edit_area"
            )

            # Render LaTeX preview if it looks like LaTeX
            if edited_text and ("^" in edited_text or "\\" in edited_text):
                try:
                    st.subheader("Rendered Equation")
                    st.latex(edited_text)
                except Exception:
                    pass

            if st.button("🧮 Solve Extracted Problem", key="solve_image_problem"):
                if edited_text.strip():
                    process_problem(
                        edited_text,
                        "image",
                        image_path=st.session_state.get("image_save_path")
                    )
                else:
                    st.warning("⚠️ Extracted text is empty. Please check the image.")


# -----------------------------
# Audio Tab
# -----------------------------

with tab3:

    uploaded_audio = st.file_uploader(
        "Upload audio problem",
        type=["wav", "mp3", "m4a"],
        key="audio_uploader"
    )

    # Clear audio session state when a new file is uploaded
    if uploaded_audio:
        prev_audio = st.session_state.get("audio_last_filename")
        if prev_audio != uploaded_audio.name:
            st.session_state.pop("audio_transcript", None)
            st.session_state.pop("audio_save_path", None)
            st.session_state["audio_last_filename"] = uploaded_audio.name

    if uploaded_audio:

        st.audio(uploaded_audio)

        # ── STEP 1: Transcribe button ──────────────────────────────
        if st.button("🎙️ Transcribe Audio", key="transcribe_audio"):

            save_path = Config.UPLOADS_DIR / f"{uuid.uuid4()}.wav"

            with open(save_path, "wb") as f:
                f.write(uploaded_audio.getbuffer())

            with st.spinner("Transcribing audio..."):
                result = get_heavy_component("audio_processor").process_audio(save_path)

            transcript = result.get("transcript", "")

            if not transcript:
                st.error("❌ Transcription failed or returned empty. Please try again.")
            else:
                st.session_state["audio_transcript"] = transcript
                st.session_state["audio_save_path"] = str(save_path)
                st.success("✅ Transcription complete!")

        # ── STEP 2: Editor and Solve (OUTSIDE transcribe block) ────
        if st.session_state.get("audio_transcript"):

            edited_text = st.text_area(
                "Transcription (edit if needed):",
                value=st.session_state["audio_transcript"],
                height=150,
                key="audio_transcript_edit"
            )

            if st.button("🧮 Solve Transcribed Problem", key="solve_audio_problem"):
                if edited_text.strip():
                    process_problem(
                        edited_text,
                        "audio",
                        audio_path=st.session_state.get("audio_save_path")
                    )
                else:
                    st.warning("⚠️ Transcription is empty. Please re-record or retype.")


# -----------------------------
# Footer
# -----------------------------

st.divider()

st.caption("Math Mentor • AI Multi-Agent Math Tutor")
