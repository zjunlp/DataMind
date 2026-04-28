from openai import OpenAI
import os
from typing import List, Dict, Optional
import base64
from pathlib import Path
import mimetypes

OPENAI_API_KEY = os.getenv("DOC_OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("DOC_OPENAI_BASE_URL")

MODEL_NAME = os.getenv("DOC_MODEL_NAME", "deepseek-chat")

def query_document(query: str, file_path: str) -> str:
    def read_document(file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    return f.read()
            except:
                return "Can't read file due to encoding issues."
        except Exception as e:
            return f"[Error]: {str(e)}"

    if file_path.lower().endswith('.md') or file_path.lower().endswith('.txt'):
        content = read_document(file_path)
    elif file_path.lower().endswith('.json') or file_path.lower().endswith('.jsonl') or file_path.lower().endswith('.csv'):
        content = read_document(file_path)
        # truncate large json/csv files if needed
        content = content[:10000] + "...(truncated)" if len(content) > 10000 else content
    else:
        return "Unsupported file format. Please provide a .txt, .md, .json, .jsonl, or .csv file."
    
    if not content or len(content.strip()) == 0:
        return "The document is empty."
    
    prompt = f"""
Please answer the following question based on the provided document content.

Document content:
{content}

User question:
{query}

Please provide an accurate and concise answer. If the document does not contain relevant information, please clearly state that.
"""
    
    try:
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a document analysis expert."},
                {"role": "user", "content": prompt.strip()}
            ],
            temperature=1.0,
            top_p=0.95,
            max_tokens=4096,
            extra_body={
                "chat_template_kwargs": {"thinking": False}
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error]: {str(e)}"