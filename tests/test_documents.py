import io

from app.retrieval.chunker import chunk_pages
from app.retrieval.loader import PageText


def _make_pdf_bytes(text: str) -> bytes:
    import fitz  # PyMuPDF

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_chunker_produces_chunks_with_metadata():
    pages = [PageText(page_number=1, text="word " * 500)]
    chunks = chunk_pages(pages, document_id="doc1", filename="f.txt", chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert all(c.document_id == "doc1" for c in chunks)
    assert all(c.filename == "f.txt" for c in chunks)


def test_upload_txt_document(client):
    data = {"file": (io.BytesIO(b"The quarterly revenue grew significantly this year."), "report.txt")}
    resp = client.post("/api/documents/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["filename"] == "report.txt"
    assert body["chunk_count"] >= 1


def test_upload_rejects_bad_extension(client):
    data = {"file": (io.BytesIO(b"not allowed"), "malware.exe")}
    resp = client.post("/api/documents/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "INVALID_FILE"


def test_upload_rejects_oversized_file(app, client):
    app.config["MAX_FILE_SIZE_MB"] = 0
    data = {"file": (io.BytesIO(b"some content"), "big.txt")}
    resp = client.post("/api/documents/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_upload_pdf_and_extract_text(client):
    pdf_bytes = _make_pdf_bytes("Annual report: major financial risks include currency exposure.")
    data = {"file": (io.BytesIO(pdf_bytes), "annual.pdf")}
    resp = client.post("/api/documents/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 201
    assert resp.get_json()["chunk_count"] >= 1


def test_list_documents(client):
    data = {"file": (io.BytesIO(b"hello world"), "notes.md")}
    client.post("/api/documents/upload", data=data, content_type="multipart/form-data")
    resp = client.get("/api/documents")
    assert resp.status_code == 200
    assert len(resp.get_json()["documents"]) >= 1


def test_document_search_returns_relevant_chunk(client):
    data = {"file": (io.BytesIO(b"The company's main financial risk is currency exchange volatility."), "risk.txt")}
    client.post("/api/documents/upload", data=data, content_type="multipart/form-data")
    resp = client.post("/api/documents/search", json={"query": "financial risk", "top_k": 3})
    assert resp.status_code == 200
    results = resp.get_json()["results"]
    assert len(results) >= 1
    assert "risk" in results[0]["text"].lower() or "financial" in results[0]["text"].lower()


def test_document_search_empty_index_returns_no_results(client):
    resp = client.post("/api/documents/search", json={"query": "anything"})
    assert resp.status_code == 200
    assert resp.get_json()["results"] == []


def test_delete_document(client):
    data = {"file": (io.BytesIO(b"temporary content"), "temp.txt")}
    upload_resp = client.post("/api/documents/upload", data=data, content_type="multipart/form-data")
    doc_id = upload_resp.get_json()["id"]
    del_resp = client.delete(f"/api/documents/{doc_id}")
    assert del_resp.status_code == 200
    list_resp = client.get("/api/documents")
    assert all(d["id"] != doc_id for d in list_resp.get_json()["documents"])


def test_document_search_tool_via_agent(client):
    data = {"file": (io.BytesIO(b"Our uploaded document explains that the main risk is supply chain disruption."), "doc.txt")}
    client.post("/api/documents/upload", data=data, content_type="multipart/form-data")
    resp = client.post(
        "/api/research",
        json={"query": "According to the uploaded document, what is the main risk?", "depth": "standard"},
    )
    assert resp.status_code == 200
