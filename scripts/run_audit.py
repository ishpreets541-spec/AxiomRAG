# run_audit.py
from app.main import main
from app.utils.llm import llm
from pprint import pprint

if __name__ == "__main__":
    query = "Does Google share user data with third parties?"
    answer = "Google may share personal information with third parties for business purposes, legal reasons, and with consent."
    result = main(query, answer, llm)
    pprint(result.dict())
