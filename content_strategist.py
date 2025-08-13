# content_strategist.py - Content Strategy and Script Generation

import json
from typing import Any, Dict, List, Optional

import ollama

from config import settings


class ContentStrategist:
    """Handles content strategy and script generation with structured JSON output."""

    def __init__(self, ollama_url: str = None, ollama_model: str = None):
        """Initialize the ContentStrategist.

        Args:
            ollama_url: Ollama server URL (defaults to config)
            ollama_model: Ollama model to use (defaults to config)
        """
        self.ollama_url = ollama_url or settings.OLLAMA_URL
        self.ollama_model = ollama_model or settings.OLLAMA_MODEL

    def write_script(
        self, video_idea: Dict[str, Any], channel_name: str
    ) -> Optional[List[Dict[str, str]]]:
        """Generate a structured script in JSON format with script_text and visual_keywords.

        Args:
            video_idea: Dictionary containing video idea information
            channel_name: Name of the YouTube channel

        Returns:
            List of scene objects, each with "script_text" and "visual_keywords", or None if failed
        """
        try:
            # Create the prompt for structured JSON output
            prompt = f"""You are a master scriptwriter for viral YouTube documentaries.

            Write a detailed script for a 10-15 minute video on: '{video_idea.get('title', 'N/A')}'

            CRITICAL REQUIREMENTS:
            - Generate 15-20 scenes/paragraphs for a 10-15 minute video
            - Each scene should be 2-3 sentences long
            - Structure the output as a JSON array of scene objects
            - Each scene object MUST have exactly two keys:
              * "script_text": The voiceover text for that scene
              * "visual_keywords": A string of 3-5 descriptive keywords for visuals (e.g., "ancient roman soldiers marching", "glowing brain neurons", "futuristic city skyline at night")

            REQUIRED JSON FORMAT:
            [
              {{
                "script_text": "The first scene voiceover text goes here. This should be 2-3 sentences that flow naturally together.",
                "visual_keywords": "ancient roman soldiers marching, battlefield smoke, dramatic sunset lighting"
              }},
              {{
                "script_text": "The second scene voiceover text continues the story. Each scene should advance the narrative.",
                "visual_keywords": "glowing brain neurons, scientific laboratory, blue light effects"
              }}
            ]

            IMPORTANT:
            - Return ONLY valid JSON array
            - Do not include any explanatory text outside the JSON
            - Each visual_keywords string should be specific and descriptive
            - Ensure the script flows logically from scene to scene
            - Make each scene engaging and visually interesting
            """

            # Get response from Ollama
            response = ollama.chat(
                model=self.ollama_model, messages=[{"role": "user", "content": prompt}]
            )

            raw_text = response.get("message", {}).get("content", "")
            if not raw_text:
                print("❌ No response received from Ollama")
                return None

            # Try to extract and parse JSON
            try:
                # Clean the response text
                cleaned_text = raw_text.strip()
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]
                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]
                cleaned_text = cleaned_text.strip()

                # Parse JSON
                script_data = json.loads(cleaned_text)

                # Validate structure
                if not isinstance(script_data, list):
                    print("❌ Response is not a list of scenes")
                    return None

                # Validate each scene has required keys
                for i, scene in enumerate(script_data):
                    if not isinstance(scene, dict):
                        print(f"❌ Scene {i} is not a dictionary")
                        return None
                    if "script_text" not in scene or "visual_keywords" not in scene:
                        print(f"❌ Scene {i} missing required keys")
                        return None

                print(f"✅ Generated script with {len(script_data)} scenes")
                return script_data

            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON response: {e}")
                print(f"Raw response: {raw_text[:200]}...")
                return None

        except Exception as e:
            print(f"❌ Error in write_script: {e}")
            return None
