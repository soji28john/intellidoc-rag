import requests

def test_query_endpoint():
    """Test the /query API endpoint"""
    response = requests.post(
        "http://localhost:8000/query",
        json={"question": "What is artificial intelligence?"}
    )
    assert response.status_code == 200, f"Unexpected status: {response.status_code}"
    data = response.json()
    assert "answer" in data, "Missing 'answer' key in response"
    assert isinstance(data["answer"], str)
