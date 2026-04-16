from __future__ import annotations

from fastapi.testclient import TestClient

from codeforge.app import app

client = TestClient(app)


class TestRoot:
    def test_root_returns_200(self) -> None:
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_name(self) -> None:
        data = client.get("/").json()
        assert data["name"] == "code-forge"

    def test_root_status(self) -> None:
        data = client.get("/").json()
        assert data["status"] == "ok"

    def test_root_version(self) -> None:
        data = client.get("/").json()
        assert data["version"] == "0.2.0"


class TestFavicon:
    def test_favicon_returns_204(self) -> None:
        resp = client.get("/favicon.ico")
        assert resp.status_code == 204


class TestListTasks:
    def test_returns_200(self) -> None:
        resp = client.get("/tasks")
        assert resp.status_code == 200

    def test_has_three_tasks(self) -> None:
        data = client.get("/tasks").json()
        assert len(data["tasks"]) == 3

    def test_has_action_schema(self) -> None:
        data = client.get("/tasks").json()
        assert "action_schema" in data

    def test_action_types_listed(self) -> None:
        data = client.get("/tasks").json()
        types = data["action_schema"]["action_types"]
        assert "submit" in types
        assert "query_kb" in types
        assert "get_audit" in types

    def test_task_has_tools_field(self) -> None:
        data = client.get("/tasks").json()
        task = data["tasks"][0]
        assert "tools" in task
        assert isinstance(task["tools"], list)


class TestHealthCheck:
    def test_returns_200(self) -> None:
        resp = client.get("/health/deep")
        assert resp.status_code == 200

    def test_has_checks(self) -> None:
        data = client.get("/health/deep").json()
        assert "checks" in data

    def test_ruff_in_checks(self) -> None:
        data = client.get("/health/deep").json()
        assert "ruff" in data["checks"]

    def test_mypy_in_checks(self) -> None:
        data = client.get("/health/deep").json()
        assert "mypy" in data["checks"]

    def test_pytest_in_checks(self) -> None:
        data = client.get("/health/deep").json()
        assert "pytest" in data["checks"]

    def test_corpus_in_checks(self) -> None:
        data = client.get("/health/deep").json()
        assert "corpus" in data["checks"]

    def test_status_field_present(self) -> None:
        data = client.get("/health/deep").json()
        assert data["status"] in ("ok", "degraded")
