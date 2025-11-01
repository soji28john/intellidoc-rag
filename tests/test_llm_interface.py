import pytest
import json
from unittest.mock import patch, MagicMock
from src.llm_interface import LLMInterface


@pytest.fixture
def llm():
    return LLMInterface(model="llama3", temperature=0.1)


@patch("requests.post")
def test_generate_response(mock_post, llm):
    # Fake Ollama response mock
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": "This is a test response"}
    }
    mock_post.return_value = mock_response

    query = "What is AI?"
    context = ["AI stands for Artificial Intelligence"]

    result = llm.generate_response(query, context)
    assert isinstance(result, str)
    assert "test response" in result.lower()


@patch("requests.post")
def test_generate_with_citations(mock_post, llm):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": "AI is discussed here [Source 1]"}
    }
    mock_post.return_value = mock_response

    query = "What is AI?"
    context = [
        {"content": "AI stands for Artificial Intelligence", "metadata": {"source": "wiki", "chunk_id": 1}}
    ]

    result = llm.generate_with_citations(query, context)

    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["sources"], list)
    assert result["confidence"] in ["low", "medium", "high"]
