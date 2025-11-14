import json
import os
from typing import Dict, Any, Optional
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
try:
    # Enable retry/backoff on the provider-level client as well
    from utils.openai_retry import wrap_openai_client
    wrap_openai_client(client)
except Exception:
    pass

def parse_response(response: str) -> Dict[str, Any]:
    """Parse JSON response from OpenAI"""
    try:
        if isinstance(response, dict):
            return response
            
        # If response contains schema definition
        if '"type": "object"' in response and '"properties":' in response:
            print("Warning: Response contains schema instead of content. Attempting to generate proper response...")
            return {"error": "Invalid response format - contains schema instead of content"}
            
        parsed = json.loads(response)
        if not isinstance(parsed, dict):
            return {"error": f"Expected dict, got {type(parsed)}"}
            
        return parsed
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return {"error": f"Failed to parse JSON: {e}"}

def generate_object(
        system_prompt: str,
        user_prompt: str,
        schema: dict,
        model: str = "gpt-4-1106-preview",  # Using GPT-4 Turbo for good balance
        temperature: float = 0.1,
        timeout: Optional[int] = None,
        max_tokens: int = 4000  # Balanced token limit
) -> str:
    """Generate structured object using OpenAI"""
    try:
        # Truncate prompts if needed to stay within context limits
        max_prompt_tokens = 24000  # Higher limit for GPT-4
        system_prompt = trim_prompt(system_prompt, max_prompt_tokens // 2)
        user_prompt = trim_prompt(user_prompt, max_prompt_tokens // 2)
        
        json_system_prompt = f"""{system_prompt}
        
        Important: Your response must be a valid JSON object containing ONLY the content, not the schema definition.
        The response should match this structure: {schema}
        
        Example if the schema asks for a 'report_markdown':
        {{"report_markdown": "# Title\\n## Section\\nContent..."}}
        
        NOT:
        {{"type": "object", "properties": {...}}}"""
        
        messages = [
            {"role": "system", "content": json_system_prompt},
            {"role": "user", "content": f"Provide a JSON response for: {user_prompt}"}
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={ "type": "json_object" }
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error in generate_object: {e}")
        if "context_length_exceeded" in str(e):
            # If context length is exceeded, try with more aggressive truncation
            try:
                system_prompt = trim_prompt(system_prompt, 4000)
                user_prompt = trim_prompt(user_prompt, 4000)
                json_system_prompt = f"{system_prompt}\nProvide your response as a JSON object that matches this schema: {schema}"
                messages = [
                    {"role": "system", "content": json_system_prompt},
                    {"role": "user", "content": f"Provide a JSON response for: {user_prompt}"}
                ]
                
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={ "type": "json_object" }
                )
                return response.choices[0].message.content
            except Exception as e2:
                print(f"Error in fallback generate_object attempt: {e2}")
                raise
        raise

def trim_prompt(text: str, max_length: int = 14000) -> str:
    """Trim text to max length while preserving complete sentences and meaning"""
    if len(text) <= max_length:
        return text
        
    # First try to trim at sentence boundary
    last_period = text.rfind('.', 0, max_length)
    if last_period != -1 and last_period > max_length * 0.7:  # Only use period if it's reasonably far
        return text[:last_period + 1].strip()
        
    # If no good sentence boundary, try paragraph
    last_newline = text.rfind('\n', 0, max_length)
    if last_newline != -1 and last_newline > max_length * 0.7:
        return text[:last_newline].strip()
        
    # If no natural breaks, preserve whole words
    last_space = text.rfind(' ', 0, max_length)
    if last_space != -1:
        return text[:last_space].strip()
        
    # Last resort: hard truncate
    return text[:max_length].strip()

    import json
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Error parsing response: {e}")
        print(f"Response was: {response}")
        return {}