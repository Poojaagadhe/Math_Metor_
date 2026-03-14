import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("No API key")
    exit(1)

client = Groq(api_key=api_key)

try:
    models = client.models.list()
    vision_models = [m.id for m in models.data if "vision" in m.id.lower()]
    all_models = [m.id for m in models.data]
    print(f"Vision models: {vision_models}")
    print(f"All models: {all_models}")
except Exception as e:
    print(f"Error: {e}")
