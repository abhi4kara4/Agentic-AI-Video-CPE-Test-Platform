import ollama
from typing import Optional, Dict, List, Any
import base64
import json
from src.config import settings
from src.utils.logger import log
import asyncio
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential


class OllamaClient:
    """Client for interacting with Ollama LLaVA model"""
    
    def __init__(self):
        self.client = ollama.Client(host=settings.ollama_host)
        self.model = settings.ollama_model
        self.timeout = settings.ollama_timeout
        
    async def check_model(self) -> bool:
        """Check if the model is available"""
        try:
            models = await self._list_models()
            return any(self.model in m['name'] for m in models)
        except Exception as e:
            log.error(f"Error checking model: {e}")
            return False
    
    async def pull_model(self) -> bool:
        """Pull the model if not available"""
        try:
            log.info(f"Pulling model {self.model}...")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{settings.ollama_host}/api/pull",
                    json={"name": self.model},
                    timeout=aiohttp.ClientTimeout(total=3600)  # 1 hour timeout for model download
                ) as response:
                    if response.status == 200:
                        async for line in response.content:
                            if line:
                                data = json.loads(line)
                                if 'status' in data:
                                    log.info(f"Pull status: {data['status']}")
                        return True
                    else:
                        log.error(f"Failed to pull model: {response.status}")
                        return False
        except Exception as e:
            log.error(f"Error pulling model: {e}")
            return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def analyze_image(self, image_base64: str, prompt: str) -> Optional[Dict[str, Any]]:
        """Analyze image with LLaVA model"""
        try:
            # Prepare the request
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_base64]
                }
            ]
            
            # Make async request to Ollama
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{settings.ollama_host}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # Low temperature for consistent results
                            "top_p": 0.9,
                            "num_predict": 1000
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "response": result["message"]["content"],
                            "model": result.get("model", self.model),
                            "total_duration": result.get("total_duration", 0) / 1e9  # Convert to seconds
                        }
                    else:
                        error_text = await response.text()
                        log.error(f"Ollama API error: {response.status} - {error_text}")
                        return None
                        
        except asyncio.TimeoutError:
            log.error(f"Timeout analyzing image after {self.timeout}s")
            return None
        except Exception as e:
            log.error(f"Error analyzing image: {e}")
            return None
    
    async def _list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{settings.ollama_host}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("models", [])
                return []
    
    def create_screen_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Create a detailed prompt for screen analysis"""
        base_prompt = """Analyze this TV/STB screen image and provide a structured response.

Current Context:
- Expected Screen: {expected_screen}
- Last Action: {last_action}
- Test Step: {test_step}

Please analyze the image and respond with the following JSON structure:
{{
    "screen_type": "Identify the type of screen (home, app_rail, app_loading, app_content, login, profile_selection, error, black_screen, etc.)",
    "detected_elements": [
        {{
            "type": "Type of element (button, text, icon, menu_item, video_player, etc.)",
            "label": "Text or label on the element",
            "position": "General position (top-left, center, bottom-right, etc.)",
            "is_focused": true/false
        }}
    ],
    "detected_text": ["List of all readable text on screen"],
    "app_name": "Name of the current app if identifiable",
    "content_playing": true/false,
    "anomalies": {{
        "black_screen": true/false,
        "frozen_frame": true/false,
        "buffering": true/false,
        "error_message": "Any error message text"
    }},
    "navigation_suggestions": {{
        "can_navigate_up": true/false,
        "can_navigate_down": true/false,
        "can_navigate_left": true/false,
        "can_navigate_right": true/false,
        "recommended_action": "Suggested next action (press_ok, navigate_down, go_back, etc.)"
    }},
    "confidence": 0.0-1.0
}}

Be precise and factual. Only report what you can clearly see in the image."""
        
        return base_prompt.format(
            expected_screen=context.get("expected_screen", "unknown"),
            last_action=context.get("last_action", "none"),
            test_step=context.get("test_step", "unknown")
        )
    
    def create_validation_prompt(self, expected_state: str) -> str:
        """Create prompt for validating expected state"""
        return f"""Analyze this TV screen and determine if it matches the expected state.

Expected State: {expected_state}

Respond with JSON:
{{
    "matches_expected": true/false,
    "actual_state": "Description of what you actually see",
    "confidence": 0.0-1.0,
    "reason": "Explanation of why it matches or doesn't match"
}}"""