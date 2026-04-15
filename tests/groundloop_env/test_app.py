from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tiny_corpus_path: Path) -> TestClient:
    monkeypatch.setenv("GROUNDLOOP_CORPUS_PATH", str(tiny_corpus_path))
    import importlib
    import groundloop_env.app as app_mod
    importlib.reload(app_mod)
    return TestClient(app_mod.app)


def test_health(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["name"] == "code-forge"


def test_tasks_endpoint(client: TestClient):
    r = client.get("/tasks")
    assert r.status_code == 200
    data = r.json()
    levels = {t["difficulty"] for t in data["tasks"]}
    assert levels == {"easy", "medium", "hard"}


def test_reset_endpoint(client: TestClient):
    r = client.post("/reset", json={"task_level": "easy"})
    assert r.status_code == 200
    body = r.json()
    obs = body.get("observation", body)
    assert obs["task_level"] == "easy"
