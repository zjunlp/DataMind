from openai import OpenAI
import datetime
import requests
import json

def Chat_With_LLM_Messages(model_name = "deepseek-ai/DeepSeek-V3", temperature=0.7 , max_tokens=1024, top_p = 1.0, messages=[{"role": "user", "content": "Hello"}], port = 8000, api_key = None):
    if "deepseek" in model_name:
        client = OpenAI(api_key = api_key, base_url="https://api.deepseek.com")
    elif "gpt" in model_name:
        client = OpenAI(api_key = api_key, base_url="https://api.openai.com/v1")
    elif "DataMind" in model_name:
        client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")

    response = client.chat.completions.create(
        model= model_name,
        messages= messages,
        stream = False,
        temperature = temperature,
        max_tokens = max_tokens,
        top_p = top_p,
        stop=None
    )
    return response.choices[0].message.content



    