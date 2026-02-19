import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src", "ceras"))

from llm_utils import check_connection, get_llm_instance
from inference import get_verifier_llm
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

def test_connection_check():
    print("Testing Connection Check (Mocked)...")
    # Real connection check would fail without valid key, but we want to fail gracefully
    
    # Test 1: Empty key returns False
    assert check_connection("Groq", "") == False
    assert check_connection("Gemini", "") == False
    print("Empty key check passed.")

    # Test 2: Invalid key returns False (assuming network/auth error caught)
    # This might take time if it actually tries to connect, so we might skip or expect failure printing
    print("Invalid key check (expect failure print):")
    res = check_connection("Groq", "invalid_key")
    assert res == False
    print("Invalid key check passed.")

def test_model_propagation():
    print("\nTesting Model Propagation...")
    
    # Test 3: Groq with specific model
    config_groq = {
        "main_provider": "Groq",
        "groq_api_key": "dummy",
        "main_model": "mixtral-8x7b-32768"
    }
    llm = get_llm_instance(config_groq["main_model"], api_config=config_groq)
    print(f"Groq Model: {llm.model_name}")
    assert llm.model_name == "mixtral-8x7b-32768"

    # Test 4: Gemini with specific model
    config_gemini = {
        "main_provider": "Gemini",
        "gemini_api_key": "dummy",
        "main_model": "gemini-1.5-pro"
    }
    llm = get_llm_instance(config_gemini["main_model"], api_config=config_gemini)
    print(f"Gemini Model: {llm.model}")
    assert llm.model == "gemini-1.5-pro"

def test_verifier_propagation():
    print("\nTesting Verifier Propagation...")

    # Test 5: Verifier specific model
    config_ver = {
        "verifier_provider": "Groq",
        "groq_api_key": "dummy",
        "verifier_model": "gemma2-9b-it" 
    }
    v_llm = get_verifier_llm(api_config=config_ver)
    print(f"Verifier Groq Model: {v_llm.model_name}")
    assert v_llm.model_name == "gemma2-9b-it"

    config_ver_gem = {
        "verifier_provider": "Gemini",
        "gemini_api_key": "dummy",
        "verifier_model": "gemini-1.5-pro"
    }
    v_llm = get_verifier_llm(api_config=config_ver_gem)
    print(f"Verifier Gemini Model: {v_llm.model}")
    assert v_llm.model == "gemini-1.5-pro"


if __name__ == "__main__":
    try:
        test_connection_check()
        test_model_propagation()
        test_verifier_propagation()
        print("\nSUCCESS: All enhancement tests passed!")
    except Exception as e:
        print(f"\nFAILURE: {e}")
        import traceback
        traceback.print_exc()
