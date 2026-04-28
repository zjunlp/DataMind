from openai import OpenAI
import os
from typing import List, Dict, Optional
import base64
from pathlib import Path
import mimetypes

OPENAI_API_KEY = os.getenv("IMG_OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("IMG_OPENAI_BASE_URL")

MODEL_NAME = os.getenv("IMG_MODEL_NAME", "Qwen3-VL-235B-A22B-Instruct")

def query_image(query: str, image_path: str) -> str:
    def encode_image(image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Image encoding failed: {str(e)}")
            return None

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    
    base64_image = encode_image(image_path)
    if base64_image is None:
        return "Can't read the image file"

    content = [
        {"type": "text", "text": query},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}"
            }
        }
    ]
    
    try:
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": content}
            ],
            temperature=0.2,
            top_p=0.95,
            max_tokens=4096
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error]: {str(e)}"