
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
key = os.environ.get("GROQ_API_KEY")
print(f"Key loaded: {bool(key)}")
if key:
    print(f"Key prefix: {key[:5]}...")

try:
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=key)
    print("Invoking LLM...")
    res = llm.invoke("Hi")
    print(f"Result: {res.content}")
except Exception as e:
    print(f"Error: {e}")
