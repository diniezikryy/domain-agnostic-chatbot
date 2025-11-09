from typing import Dict, Any, Optional
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_object(
    system_prompt: str,
    user_prompt: str,
    schema: Dict[str, Any],
    model: str = "gpt-4-1106-preview",
    temperature: float = 0.1,
    timeout: Optional[int] = None
) -> str:
    """Generate structured object using OpenAI"""
    try:
        json_system_prompt = f"{system_prompt}\nProvide your response as a JSON object that matches this schema: {schema}"
        messages = [
            {"role": "system", "content": json_system_prompt},
            {"role": "user", "content": f"Provide a JSON response for: {user_prompt}"}
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={ "type": "json_object" }
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error in generate_object: {e}")
        raise

def trim_prompt(text: str, max_length: int = 14000) -> str:
    """Trim text to max length while preserving complete sentences"""
    if len(text) <= max_length:
        return text
        
    # Find the last period before max_length
    last_period = text.rfind('.', 0, max_length)
    if last_period == -1:
        return text[:max_length]
        
    return text[:last_period + 1]

def parse_response(response: str) -> Dict[str, Any]:
    """Parse JSON response from OpenAI"""
    import json
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Error parsing response: {e}")
        print(f"Response was: {response}")
        return {}