import requests

# Test query endpoint
response = requests.post(
    "http://localhost:8000/query",
    json={
        "question": "What is the main topic?",
        "n_results": 5,
        "include_sources": True
    }
)


print(response.json())