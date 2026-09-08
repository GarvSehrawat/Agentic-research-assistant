def test_missing_query_returns_400(client):
    resp = client.post("/api/research", json={"depth": "basic"})
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "INVALID_REQUEST"


def test_empty_query_returns_400(client):
    resp = client.post("/api/research", json={"query": "   "})
    assert resp.status_code == 400


def test_invalid_depth_returns_400(client):
    resp = client.post("/api/research", json={"query": "What is Python?", "depth": "extreme"})
    assert resp.status_code == 400


def test_simple_query_completes(client):
    resp = client.post("/api/research", json={"query": "What is machine learning?", "depth": "basic"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "completed"
    assert data["answer"]
    assert data["tasks"] == []  # simple queries skip planning


def test_calculator_query_routes_to_calculator(client):
    resp = client.post("/api/research", json={"query": "Calculate 25 * 18", "depth": "basic"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert any(t["assigned_tool"] == "calculator" for t in data["tasks"])
    calc_task = next(t for t in data["tasks"] if t["assigned_tool"] == "calculator")
    assert calc_task["result"] == "450.0" or calc_task["result"] == "450"


def test_web_research_query_fails_gracefully_without_key(client):
    resp = client.post("/api/research", json={"query": "What are the latest developments in EVs?"})
    assert resp.status_code == 200
    data = resp.get_json()
    # No WEB_SEARCH_API_KEY configured in the test app -> tool fails, but
    # the request as a whole still completes with a controlled answer.
    assert data["status"] in ("completed", "failed")
    web_tasks = [t for t in data["tasks"] if t["assigned_tool"] == "web_search"]
    assert web_tasks
    assert web_tasks[0]["status"] == "FAILED"


def test_get_and_cancel_session(client):
    create_resp = client.post("/api/research", json={"query": "What is Python?"})
    session_id = create_resp.get_json()["session_id"]

    get_resp = client.get(f"/api/research/{session_id}")
    assert get_resp.status_code == 200
    assert get_resp.get_json()["session_id"] == session_id

    # Already completed -> cancelling should be rejected.
    cancel_resp = client.post(f"/api/research/{session_id}/cancel")
    assert cancel_resp.status_code == 409


def test_get_missing_session_404(client):
    resp = client.get("/api/research/does-not-exist")
    assert resp.status_code == 404


def test_llm_unavailable_returns_503(app, client):
    app.extensions["llm_provider"] = None
    resp = client.post("/api/research", json={"query": "What is Python?"})
    assert resp.status_code == 503
    assert resp.get_json()["error"]["code"] == "LLM_UNAVAILABLE"


def test_max_tool_calls_limit_is_enforced(app, client):
    app.config["MAX_TOOL_CALLS"] = 0
    resp = client.post("/api/research", json={"query": "Calculate 2 * 2"})
    data = resp.get_json()
    calc_task = next(t for t in data["tasks"] if "2" in t["description"])
    assert calc_task["status"] == "FAILED"
    assert "MAX_TOOL_CALLS" in calc_task["result"]
