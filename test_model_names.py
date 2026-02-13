
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

models_to_test = [
    "gpt-oss-20b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768"
]

print(f"API Key loaded: {bool(api_key)}")

for model in models_to_test:
    print(f"\n--- Testing Model: {model} ---")
    try:
        llm = ChatGroq(model=model, api_key=api_key)
        res = llm.invoke("Hello, are you working?")
        print(f"SUCCESS. Response: {res.content}")
    except Exception as e:
        print(f"FAILED. Error: {e}")
