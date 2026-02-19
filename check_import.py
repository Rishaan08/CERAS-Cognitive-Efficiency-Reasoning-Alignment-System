try:
    import langchain_google_genai
    print("SUCCESS: langchain_google_genai imported")
except ImportError as e:
    print(f"FAILURE: {e}")
except Exception as e:
    print(f"ERROR: {e}")
