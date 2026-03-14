"""
Base agent class for all agents
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import os
import streamlit as st

from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self, name: str, model: str, temperature: float = None):

        self.name = name
        self.model = model
        self.temperature = temperature or Config.AGENT_TEMPERATURE
        self.llm_provider = getattr(Config, "LLM_PROVIDER", "groq").lower()

        logger.info(f"Initializing {self.name} with provider: {self.llm_provider}")

        if self.llm_provider == "gemini":
            self._init_gemini()

        elif self.llm_provider == "groq":
            self._init_groq()

        elif self.llm_provider == "huggingface":
            self._init_huggingface()

        elif self.llm_provider == "ollama":
            self._init_ollama()

        else:
            self._init_openai()

    # -----------------------------------------------------
    # API KEY RESOLVER
    # -----------------------------------------------------

    def _get_api_key(self, key_name: str):
        """Resolve API key from Streamlit secrets, env, or Config"""

        if key_name in st.secrets:
            return st.secrets[key_name]

        if os.getenv(key_name):
            return os.getenv(key_name)

        return getattr(Config, key_name, None)

    # -----------------------------------------------------
    # PROVIDER INITIALIZATION
    # -----------------------------------------------------

    def _init_groq(self):
        from groq import Groq

        api_key = self._get_api_key("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY not found")

        self.groq_client = Groq(api_key=api_key)

        logger.info(f"{self.name} initialized with Groq model {self.model}")

    def _init_gemini(self):
        import google.generativeai as genai

        api_key = self._get_api_key("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")

        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel(self.model)

        logger.info(f"{self.name} initialized with Gemini model {self.model}")

    def _init_huggingface(self):
        from huggingface_hub import InferenceClient

        api_key = self._get_api_key("HUGGINGFACE_API_KEY")

        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY not found")

        self.hf_client = InferenceClient(token=api_key)

        logger.info(f"{self.name} initialized with HF model {self.model}")

    def _init_ollama(self):
        import ollama

        self.ollama_client = ollama
        self.ollama_base_url = getattr(Config, "OLLAMA_BASE_URL", "http://localhost:11434")

        logger.info(f"{self.name} initialized with Ollama model {self.model}")

    def _init_openai(self):
        import openai

        api_key = self._get_api_key("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")

        openai.api_key = api_key

        self.openai_client = openai

        logger.info(f"{self.name} initialized with OpenAI model {self.model}")

    # -----------------------------------------------------
    # ABSTRACT RUN METHOD
    # -----------------------------------------------------

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    # -----------------------------------------------------
    # LLM CALL
    # -----------------------------------------------------

    def _call_llm(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:

        try:

            if self.llm_provider == "groq":

                response = self.groq_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or 1000,
                )

                return response.choices[0].message.content

            elif self.llm_provider == "gemini":

                prompt = self._messages_to_prompt(messages)

                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature or self.temperature,
                        "max_output_tokens": max_tokens or 1000,
                    },
                )

                return response.text

            elif self.llm_provider == "huggingface":

                prompt = self._messages_to_prompt(messages)

                result = self.hf_client.text_generation(
                    prompt,
                    max_new_tokens=max_tokens or 1000,
                    temperature=temperature or self.temperature,
                )

                return result

            elif self.llm_provider == "ollama":

                response = self.ollama_client.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": temperature or self.temperature,
                        "num_predict": max_tokens or -1,
                    },
                )

                return response["message"]["content"]

            else:

                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens,
                )

                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    # -----------------------------------------------------
    # MESSAGE UTILITIES
    # -----------------------------------------------------

    def _messages_to_prompt(self, messages: list) -> str:

        prompt = ""

        for msg in messages:

            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                prompt += f"{content}\n\n"

            elif role == "user":
                prompt += f"User: {content}\n\n"

            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"

        return prompt.strip()

    def _create_system_message(self, content: str) -> Dict[str, str]:
        return {"role": "system", "content": content}

    def _create_user_message(self, content: str) -> Dict[str, str]:
        return {"role": "user", "content": content}

    def _create_assistant_message(self, content: str) -> Dict[str, str]:
        return {"role": "assistant", "content": content}