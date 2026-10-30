"""Regression checks for persistence, offline lookup and concurrent changes.

The synthetic fixtures exercise invariants, not claims of retrieval accuracy.
"""
import json
from pathlib import Path
import threading
import time

import pytest

from zoom_copilot import knowledge
from zoom_copilot.knowledge import KnowledgeBase, Passage, chunks, read_document, tokenize


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_old_index_loads_with_stable_ids(tmp_path):
    target = tmp_path / "index.json"
    target.write_text(json.dumps({"documents": ["a.pdf"], "passages": [
        {"source": "a.pdf", "location": "第 4 页", "text": "A calibrated Stokes instrument matrix."}
    ]}), encoding="utf-8")
    kb = KnowledgeBase(target)
    passage = kb.search("Stokes instrument matrix")[0]
    assert passage.location == "第 4 页" and passage.passage_id
    assert KnowledgeBase(target).passages[0].passage_id == passage.passage_id
    assert kb.passages[0].score == 0.0  # Search must not modify stored evidence.


def test_same_path_refresh_uses_content_hash(tmp_path):
    kb = KnowledgeBase(tmp_path / "index.json")
    source = tmp_path / "notes.txt"
    write(source, "Airflow calibration is retained unchanged across wind speeds.")
    assert kb.add_files([str(source)])[0] == 1
    first_id = kb.passages[0].passage_id
    assert kb.add_files([str(source)])[0] == 0
    write(source, "Underwater degradation includes attenuation and additive background scattering.")
    assert kb.add_files([str(source)])[0] == 1
    assert len(kb.passages) == 1 and len(kb.document_names) == 1
    assert kb.passages[0].passage_id != first_id
    assert not kb.search("airflow")
    assert kb.search("underwater attenuation")
    assert kb.document_records[str(source.resolve())]["sha256"]


def test_chinese_aliases_retrieve_english_evidence(tmp_path):
    kb = KnowledgeBase(tmp_path / "index.json")
    kb.add_files([write(tmp_path / "sensing.txt", "The sensing operator maps spectral scene responses to compressed detector measurements."),
                  write(tmp_path / "airflow.txt", "No-flow calibration stays fixed and unchanged across airflow conditions.")])
    assert Path(kb.search("气流扰动下为什么保留无风标定？", 1)[0].source).name == "airflow.txt"
    assert Path(kb.search("感知算子与探测器域测量有什么关系？", 1)[0].source).name == "sensing.txt"
    assert kb.retrieval_mode == "bm25-local-terminology"


def test_unrelated_questions_do_not_force_top_k(tmp_path):
    kb = KnowledgeBase(tmp_path / "index.json")
    kb.add_files([write(tmp_path / "notes.txt", "The spectral polarization camera uses a calibrated GISC sensing operator.")])
    assert kb.search("What is the weather in London tomorrow?") == []
    assert kb.search("今天晚饭如何做红烧肉？") == []
    assert kb.search("Please tell me about your research methods.") == []
    assert kb.search("Stokes", top_k=0) == []


def test_stokes_calibration_expands_to_forward_and_inverse_terms_without_file_prior(tmp_path):
    kb = KnowledgeBase(tmp_path / "index.json")
    kb.add_files([
        write(tmp_path / "encoding.txt", "Spectral polarimetric GISC uses a calibrated sensing operator for compressed detector measurements."),
        write(tmp_path / "forward.txt", "An instrument matrix maps incident linear Stokes quantities to reconstructed analyzer intensities."),
        write(tmp_path / "inverse.txt", "The demodulation operator is the Moore Penrose pseudoinverse of the instrument matrix. It retrieves linear Stokes values after channel reconstruction."),
    ])
    found = kb.search("How do you distinguish the calibrated sensing operator from the Stokes calibration matrix?", top_k=3)
    assert any("pseudoinverse" in passage.text for passage in found)
    assert "pseudoinverse" not in knowledge._query_weights("What does the spectral sensing operator encode?")
    chinese = knowledge._query_weights("斯托克斯标定矩阵与感知算子有何区别？")
    assert chinese["demodulation"] >= .8


def test_clear_serializes_with_slow_import_but_search_remains_available(tmp_path, monkeypatch):
    kb = KnowledgeBase(tmp_path / "index.json")
    kb.add_files([write(tmp_path / "old.txt", "Old underwater measurements contain detector-domain degradation.")])
    source = write(tmp_path / "new.txt", "New spectral polarization acquisition and calibration information.")
    started, proceed = threading.Event(), threading.Event()
    original = knowledge.read_document

    def slow_read(path):
        if path.name == "new.txt":
            started.set()
            assert proceed.wait(5)
        return original(path)

    monkeypatch.setattr(knowledge, "read_document", slow_read)
    importer = threading.Thread(target=kb.add_files, args=([source],))
    importer.start()
    assert started.wait(3)
    assert kb.search("underwater")  # Immutable prior snapshot remains searchable.
    clearer = threading.Thread(target=kb.clear)
    clearer.start()
    proceed.set()
    importer.join(5)
    clearer.join(5)
    assert not importer.is_alive() and not clearer.is_alive()
    assert kb.passages == [] and kb.document_names == []
    assert KnowledgeBase(kb.index_path).passages == []


def test_concurrent_search_during_refresh(tmp_path):
    kb = KnowledgeBase(tmp_path / "index.json")
    source = tmp_path / "notes.txt"
    write(source, "Calibrated Stokes instrument matrix with spectral response.")
    kb.add_files([str(source)])
    errors = []

    def reader():
        try:
            for _ in range(80):
                for p in kb.search("Stokes calibration"):
                    assert p.source == str(source.resolve()) and p.passage_id
        except Exception as ex:
            errors.append(ex)

    worker = threading.Thread(target=reader)
    worker.start()
    for i in range(4):
        write(source, f"Calibrated Stokes instrument matrix iteration {i} with spectral response.")
        kb.add_files([str(source)])
    worker.join(5)
    assert not errors and not worker.is_alive()


def test_optional_local_hybrid_and_explicit_fallback(tmp_path):
    kb = KnowledgeBase(tmp_path / "index.json")
    kb.add_files([write(tmp_path / "notes.txt", "Underwater scattering attenuates spectral measurements in the detector domain.")])
    kb.set_embedder(lambda texts: [[1.0, 0.0] for _ in texts], "unit-test-local")
    assert kb.retrieval_mode == "hybrid-local"
    assert kb.search("underwater scattering")

    def unavailable(_):
        raise RuntimeError("local model not installed")

    kb.set_embedder(unavailable)
    assert kb.retrieval_mode == "bm25-local-terminology"
    assert "回退" in kb.embedding_warning
    assert kb.search("underwater scattering")


def test_page_boundaries_preserved_and_invalid_chunk_config_rejected(tmp_path, monkeypatch):
    kb = KnowledgeBase(tmp_path / "index.json")
    source = tmp_path / "placeholder.pdf"
    source.write_bytes(b"test-reader-placeholder")
    monkeypatch.setattr(knowledge, "read_document", lambda _: [
        ("第 1 页", "Page one sensing operator maps spectral measurements. " * 60),
        ("第 2 页", "Page two Stokes calibration uses a separate instrument matrix. " * 60)])
    kb.add_files([str(source)])
    assert len(kb.passages) > 2
    assert all(not ("Page one" in p.text and "Page two" in p.text) for p in kb.passages)
    assert {p.location for p in kb.passages} == {"第 1 页", "第 2 页"}
    with pytest.raises(ValueError):
        list(chunks("text", maximum=100, overlap=100))


def test_broken_index_is_preserved_with_warning(tmp_path):
    target = tmp_path / "index.json"
    target.write_text("{broken", encoding="utf-8")
    kb = KnowledgeBase(target)
    assert kb.load_warning and not kb.passages
    assert target.read_text(encoding="utf-8") == "{broken"


def test_document_tables_and_slide_notes(tmp_path):
    docx = pytest.importorskip("docx")
    pptx = pytest.importorskip("pptx")
    document = docx.Document()
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Detector domain"
    table.cell(0, 1).text = "Calibrated sensing operator"
    document.save(tmp_path / "table.docx")
    assert "Calibrated sensing" in read_document(tmp_path / "table.docx")[0][1]
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    slide.notes_slide.notes_text_frame.text = "Personal contribution needs user confirmation."
    presentation.save(tmp_path / "notes.pptx")
    result = read_document(tmp_path / "notes.pptx")
    assert any("讲者备注" in label and "contribution" in text for label, text in result)


def test_atomic_save_leaves_no_temp_files(tmp_path):
    kb = KnowledgeBase(tmp_path / "index.json")
    kb.add_files([write(tmp_path / "notes.txt", "Spectral polarization calibration is stored atomically in a local index.")])
    assert json.loads(kb.index_path.read_text(encoding="utf-8"))["version"] == 2
    assert not list(tmp_path.glob("*.tmp"))
    kb.clear()
    assert not json.loads(kb.index_path.read_text(encoding="utf-8"))["passages"]


def test_tokenization_handles_pdf_hyphenation_and_terms():
    assert "polarization" in tokenize("Polari-\nzation")
    assert "matrix" in tokenize("instrument matrices")
    assert "spectral" in tokenize("spectra")
    assert "the" not in tokenize("the sensing operator")
    assert "noflow" in tokenize("no-flow calibration")


def test_pdf_native_column_order_preserves_sentences(tmp_path):
    from public_fixtures import write_synthetic_pdf
    path = tmp_path / "two_columns.pdf"
    write_synthetic_pdf(path, [
        (30, 732, "Calibrated sensing operator\nencodes compressed measurements."),
        (310, 732, "A separate instrument matrix\nretrieves Stokes quantities.")])
    text = read_document(path)[0][1]
    assert "Calibrated sensing operator\nencodes compressed measurements." in text
    assert "A separate instrument matrix\nretrieves Stokes quantities." in text


def test_context_neighbours_do_not_cross_pages(tmp_path, monkeypatch):
    source = tmp_path / "placeholder.pdf"
    source.write_bytes(b"reader-fixture")
    monkeypatch.setattr(knowledge, "read_document", lambda _: [
        ("第 1 页", "Leading context has unrelated filler details. " * 30 +
         "Uniqueanchor explains a scientific mechanism. " + "Additional implications after the mechanism. " * 25),
        ("第 2 页", "PAGE_TWO_MUST_NOT_LEAK " * 30)])
    kb = KnowledgeBase(tmp_path / "index.json")
    kb.add_files([str(source)])
    found = kb.search("Uniqueanchor", top_k=1)[0]
    assert found.location == "第 1 页"
    assert "Additional implications" in found.text
    assert "PAGE_TWO_MUST_NOT_LEAK" not in found.text
    assert found.passage_id == kb.search("Uniqueanchor", top_k=1)[0].passage_id


def test_failed_disk_commit_preserves_prior_searchable_state(tmp_path, monkeypatch):
    kb = KnowledgeBase(tmp_path / "index.json")
    kb.add_files([write(tmp_path / "old.txt", "Underwater attenuation and scattering are modeled in the scene domain.")])

    def denied():
        raise PermissionError("simulated disk permission failure")

    monkeypatch.setattr(kb, "_save", denied)
    added, issues = kb.add_files([write(tmp_path / "new.txt", "Spectral instrument calibration maps Stokes vectors to intensities.")])
    assert added == 0 and issues
    assert len(kb.document_names) == 1 and kb.search("underwater")
    assert not kb.search("Stokes")
    with pytest.raises(PermissionError):
        kb.clear()
    assert kb.search("underwater")
