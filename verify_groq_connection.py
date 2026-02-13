
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src', 'ceras'))

# Load environment variables
load_dotenv()

def verify_llm_utils():
    print("Testing llm_utils (Main Model)...")
    try:
        from llm_utils import call_llm
        response = call_llm("Return the word 'HELLO' only.")
        print(f"Response: {response}")
        if "HELLO" in response:
            print("SUCCESS: llm_utils operational.")
        else:
            print("WARNING: llm_utils returned unexpected output.")
    except Exception as e:
        print(f"ERROR: llm_utils failed: {e}")

def verify_inference():
    print("\nTesting inference (Verifier Model)...")
    try:
        # We need to test the verifier. 
        # fast_verify_subtasks uses llm_verify.
        from inference import llm_verify
        response = llm_verify.invoke("Return the word 'WORLD' only.")
        content = response.content if hasattr(response, "content") else str(response)
        print(f"Response: {content}")
        if "WORLD" in content:
            print("SUCCESS: inference verifier operational.")
        else:
            print("WARNING: inference verifier returned unexpected output.")
    except Exception as e:
        print(f"ERROR: inference verifier failed: {e}")

def verify_camre_edu():
    print("\nTesting CAMRE_EDU (Verifier Model usage)...")
    try:
        from CAMRE_EDU import OLLAMA_MODEL
        print(f"CAMRE_EDU Model: {OLLAMA_MODEL}")
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=OLLAMA_MODEL, api_key=os.environ.get("GROQ_API_KEY"))
        response = llm.invoke("Say 'TEST'")
        print(f"Response: {response.content}")
        if "TEST" in response.content:
             print("SUCCESS: CAMRE_EDU model instantiation operational.")
    except Exception as e:
        print(f"ERROR: CAMRE_EDU verification failed: {e}")

if __name__ == "__main__":
    verify_llm_utils()
    verify_inference()
    verify_camre_edu()
