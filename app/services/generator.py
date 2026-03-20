# app/services/generator.py

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "Missing OPENAI_API_KEY. Add it to a .env file in the project root or export it in your shell."
    )

client = OpenAI(api_key=api_key)

def generate_answer(query, context):
    prompt = f"""
Answer based only on the context below.

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content