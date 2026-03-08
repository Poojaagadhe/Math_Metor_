"""Base agent class for all agents"""
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import openai
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, name: str, model: str, temperature: float = None):
        """
        Initialize base agent
        
        Args:
            name: Agent name
            model: LLM model to use
            temperature: Temperature for generation (uses Config default if None)
        """
        self.name = name
        self.model = model
        self.temperature = temperature or Config.AGENT_TEMPERATURE
        self.llm_provider = getattr(Config, 'LLM_PROVIDER', 'groq').lower()
        
        # Initialize based on provider
        if self.llm_provider == 'gemini':
            try:
                import google.generativeai as genai
                gemini_key = getattr(Config, 'GEMINI_API_KEY', None)
                if not gemini_key:
                    raise ValueError("GEMINI_API_KEY not found in environment")
                genai.configure(api_key=gemini_key)
                self.gemini_model = genai.GenerativeModel(self.model)
                logger.info(f"{self.name} initialized with Gemini model {self.model}")
            except ImportError:
                logger.error("google-generativeai package not installed. Run: pip install google-generativeai")
                raise
        elif self.llm_provider == 'groq':
            try:
                from groq import Groq
                groq_key = getattr(Config, 'GROQ_API_KEY', None)
                if not groq_key:
                    raise ValueError("GROQ_API_KEY not found in environment")
                self.groq_client = Groq(api_key=groq_key)
                logger.info(f"{self.name} initialized with Groq model {self.model}")
            except ImportError:
                logger.error("groq package not installed. Run: pip install groq")
                raise
        elif self.llm_provider == 'huggingface':
            try:
                from huggingface_hub import InferenceClient
                hf_token = getattr(Config, 'HUGGINGFACE_API_KEY', None)
                if not hf_token:
                    raise ValueError("HUGGINGFACE_API_KEY not found in environment")
                self.hf_client = InferenceClient(token=hf_token)
                logger.info(f"{self.name} initialized with Hugging Face model {self.model}")
            except ImportError:
                logger.error("huggingface_hub package not installed. Run: pip install huggingface_hub")
                raise
        elif self.llm_provider == 'ollama':
            try:
                import ollama
                self.ollama_client = ollama
                self.ollama_base_url = getattr(Config, 'OLLAMA_BASE_URL', 'http://localhost:11434')
                logger.info(f"{self.name} initialized with Ollama model {self.model}")
            except ImportError:
                logger.error("Ollama package not installed. Run: pip install ollama")
                raise
        else:
            # OpenAI fallback
            if not Config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not found in environment")
            openai.api_key = Config.OPENAI_API_KEY
            logger.info(f"{self.name} initialized with OpenAI model {self.model}")
        
    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Agent output
        """
        pass
        
    def _call_llm(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Call LLM with messages
        
        Args:
            messages: List of message dicts
            temperature: Override temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLM response text
        """
        try:
            if self.llm_provider == 'gemini':
                # Call Gemini API
                # Convert messages to Gemini format
                prompt = self._messages_to_gemini_prompt(messages)
                
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': temperature or self.temperature,
                        'max_output_tokens': max_tokens or 1000,
                    }
                )
                return response.text
            elif self.llm_provider == 'groq':
                # Call Groq API
                response = self.groq_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or 1000
                )
                return response.choices[0].message.content
            elif self.llm_provider == 'huggingface':
                # Call Hugging Face using direct API (no provider needed)
                import requests
                import json
                
                API_URL = f"https://api-inference.huggingface.co/models/{self.model}"
                headers = {"Authorization": f"Bearer {Config.HUGGINGFACE_API_KEY}"}
                
                # Convert messages to prompt
                prompt = self._messages_to_prompt(messages)
                
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens or 1000,
                        "temperature": temperature or self.temperature,
                        "return_full_text": False
                    }
                }
                
                response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                
                # Handle model loading (503 error)
                if response.status_code == 503:
                    result = response.json()
                    if 'estimated_time' in result:
                        logger.warning(f"Model is loading, estimated time: {result['estimated_time']}s")
                        import time
                        time.sleep(min(result['estimated_time'], 20))  # Wait max 20 seconds
                        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"HF API response type: {type(result)}, content: {str(result)[:200]}")
                
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    generated = result[0].get('generated_text', '')
                    # Remove the prompt from response if it's included
                    if generated.startswith(prompt):
                        generated = generated[len(prompt):].strip()
                    return generated
                elif isinstance(result, dict):
                    return result.get('generated_text', result.get('text', str(result)))
                else:
                    return str(result)
            elif self.llm_provider == 'ollama':
                # Call Ollama
                response = self.ollama_client.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        'temperature': temperature or self.temperature,
                        'num_predict': max_tokens or -1
                    }
                )
                return response['message']['content']
            else:
                # Call OpenAI
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _messages_to_gemini_prompt(self, messages: list) -> str:
        """Convert OpenAI-style messages to Gemini prompt format"""
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"Instructions: {content}\n")
            elif role == 'user':
                prompt_parts.append(f"User: {content}\n")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}\n")
        
        return "\n".join(prompt_parts)
    
    def _messages_to_prompt(self, messages: list) -> str:
        """Convert messages to simple prompt (for HF and others)"""
        prompt = ""
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'system':
                prompt += f"{content}\n\n"
            elif role == 'user':
                prompt += f"User: {content}\n\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n\n"
        
        return prompt.strip()
    
    def _create_system_message(self, content: str) -> Dict[str, str]:
        """Create system message"""
        return {"role": "system", "content": content}
    
    def _create_user_message(self, content: str) -> Dict[str, str]:
        """Create user message"""
        return {"role": "user", "content": content}
    
    def _create_assistant_message(self, content: str) -> Dict[str, str]:
        """Create assistant message"""
        return {"role": "assistant", "content": content}
