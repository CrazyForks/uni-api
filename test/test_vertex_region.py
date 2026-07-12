import asyncio

from core.models import RequestModel
from uni_api.providers.payloads import get_vertex_claude_payload, get_vertex_gemini_payload


def _request(model):
    return RequestModel(model=model, messages=[{"role": "user", "content": "hello"}], stream=False)


def test_vertex_gemini_defaults_to_global_region():
    provider = {
        "provider": "vertex",
        "base_url": "https://aiplatform.googleapis.com",
        "project_id": "test-project",
        "model": ["gemini-3.5-flash"],
    }

    url, _, _ = asyncio.run(get_vertex_gemini_payload(_request("gemini-3.5-flash"), "vertex-gemini", provider))

    assert url == (
        "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/"
        "publishers/google/models/gemini-3.5-flash:generateContent"
    )


def test_vertex_gemini_uses_explicit_region():
    provider = {
        "provider": "vertex",
        "base_url": "https://aiplatform.googleapis.com",
        "project_id": "test-project",
        "region": "us-central1",
        "model": ["gemini-3.5-flash"],
    }

    url, _, _ = asyncio.run(get_vertex_gemini_payload(_request("gemini-3.5-flash"), "vertex-gemini", provider))

    assert url == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/"
        "publishers/google/models/gemini-3.5-flash:generateContent"
    )


def test_vertex_claude_uses_same_region_setting():
    provider = {
        "provider": "vertex",
        "base_url": "https://aiplatform.googleapis.com",
        "project_id": "test-project",
        "region": "europe-west1",
        "model": ["claude-sonnet-4-5@20250929"],
    }

    url, _, _ = asyncio.run(get_vertex_claude_payload(_request("claude-sonnet-4-5@20250929"), "vertex-claude", provider))

    assert url == (
        "https://europe-west1-aiplatform.googleapis.com/v1/projects/test-project/locations/europe-west1/"
        "publishers/anthropic/models/claude-sonnet-4-5@20250929:streamRawPredict"
    )
