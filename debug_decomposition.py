
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.getcwd(), 'src', 'ceras'))
load_dotenv()

from llm_utils import call_llm, DECOMP_PROMPT_JSON, DECOMP_PROMPT_SIMPLE

def debug_decomposition():
    query = "Calculate (51^2 - 49^2) using algebraic identities."
    print(f"--- Query: {query} ---")

    # Test JSON Prompt
    print("\n--- Testing JSON Prompt ---")
    prompt_json = DECOMP_PROMPT_JSON.format(query=query)
    try:
        raw_json = call_llm(prompt_json)
        print(f"RAW JSON OUTPUT:\n{raw_json}")
    except Exception as e:
        print(f"JSON Prompt Failed: {e}")

    # Test Simple Prompt
    print("\n--- Testing Simple Prompt ---")
    prompt_simple = DECOMP_PROMPT_SIMPLE.format(query=query)
    try:
        raw_simple = call_llm(prompt_simple)
        print(f"RAW SIMPLE OUTPUT:\n{raw_simple}")
    except Exception as e:
        print(f"Simple Prompt Failed: {e}")

if __name__ == "__main__":
    debug_decomposition()
