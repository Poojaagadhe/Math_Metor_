import os
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("No API key")
    exit(1)

client = Groq(api_key=api_key)

# Create a tiny 1x1 white pixel image in base64
image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

models = ["meta-llama/llama-4-scout-17b-16e-instruct"]

for model in models:
    print(f"Testing model {model}...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "What is this?"
                        }
                    ]
                }
            ],
            max_tokens=10
        )
        print(f"Success! Model {model} response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error for {model}: {e}")

