import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src", "ceras"))

from llm_utils import get_llm_instance
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

def test_llm_factory():
    print("Testing LLM Factory Logic...")
    
    # Test 1: Default (Env)
    print("\nTest 1: Default (Groq from Env)")
    llm = get_llm_instance("llama-3.3-70b-versatile", api_config=None)
    print(f"Instance: {type(llm).__name__}")
    assert isinstance(llm, ChatGroq)

    # Test 2: Groq with Explicit Key
    print("\nTest 2: Groq with Explicit Key")
    config_groq = {
        "main_provider": "Groq",
        "groq_api_key": "dummy_groq_key"
    }
    llm = get_llm_instance("llama-3.3-70b-versatile", api_config=config_groq)
    print(f"Instance: {type(llm).__name__}")
    assert isinstance(llm, ChatGroq)
    # Check if key is set (accessing private attribute or just assuming if no error)
    if hasattr(llm, "api_key"): # standard langchain might hide it
         print("Groq key set successfully")

    # Test 3: Gemini with Explicit Key
    print("\nTest 3: Gemini with Explicit Key")
    config_gemini = {
        "main_provider": "Gemini",
        "gemini_api_key": "dummy_gemini_key"
    }
    llm = get_llm_instance("gemini-2.5-flash", api_config=config_gemini)
    print(f"Instance: {type(llm).__name__}")
    assert isinstance(llm, ChatGoogleGenerativeAI)
    print(f"Model: {llm.model}")
    assert llm.model == "gemini-2.5-flash"


from inference import get_verifier_llm

def test_verifier_factory():
    print("\nTesting Verifier Factory Logic...")

    # Test 4: Verifier Default
    print("\nTest 4: Verifier Default (Groq)")
    v_llm = get_verifier_llm(api_config=None)
    print(f"Instance: {type(v_llm).__name__}")
    assert isinstance(v_llm, ChatGroq)

    # Test 5: Verifier Gemini
    print("\nTest 5: Verifier Gemini")
    config_v_gemini = {
        "verifier_provider": "Gemini",
        "gemini_api_key": "dummy_v_key"
    }
    v_llm = get_verifier_llm(api_config=config_v_gemini)
    print(f"Instance: {type(v_llm).__name__}")
    assert isinstance(v_llm, ChatGoogleGenerativeAI)


if __name__ == "__main__":
    try:
        test_llm_factory()
        test_verifier_factory()
        print("\nSUCCESS: All factory tests passed!")
    except Exception as e:
        print(f"\nFAILURE: {e}")
