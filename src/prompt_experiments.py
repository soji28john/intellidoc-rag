import subprocess

# Helper function to call Ollama
def ask_ollama(prompt, question, model="llama3"):
    full_prompt = f"{prompt}\n\nUser question: {question}"
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=full_prompt,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    return result.stdout.strip()

#  Prompts to test
prompt1 = "Answer the question based only on the context provided."

prompt2 = """You are an expert assistant.
Follow rules:
1. Only use context
2. If answer not present, say 'Not in context'
3. Cite source as [Context]
"""

prompt3 = """You are a helpful assistant. Example:
Q: What is RAG?
Context: RAG means Retrieval Augmented Generation
A: According to the context, RAG means Retrieval Augmented Generation [Context].

Now answer using the same style.
"""

#  Question to test
question = "What is RAG?"

#  Run each prompt
print("\n Basic Prompt ")
print(ask_ollama(prompt1, question))

print("\n Detailed Prompt ")
print(ask_ollama(prompt2, question))

print("\n Few-shot Prompt ")
print(ask_ollama(prompt3, question))
