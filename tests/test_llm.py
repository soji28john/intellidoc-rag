from src.llm_interface import LLMInterface

llm = LLMInterface()

context = [
    "Python was created by Guido van Rossum in 1991."
]

query = "Who created Python?"

response = llm.generate_response(query, context)
print("\nResponse:", response)
