"""
One-off script: verify Groq API is reachable and the key works.
Run once during setup. Delete after Phase 0 if you like.
"""
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key or api_key.startswith("TODO"):
    raise RuntimeError("GROQ_API_KEY not set in .env. Fix step 0.9.")

model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
client = Groq(api_key=api_key)

resp = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Say 'RootAI setup verified' and nothing else."},
    ],
    max_tokens=20,
)

print("MODEL:", model)
print("RESPONSE:", resp.choices[0].message.content)
print("TOKENS:", resp.usage.total_tokens)
print("OK")