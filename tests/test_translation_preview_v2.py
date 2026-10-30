"""Concurrency/priority tests only; real translation quality is measured separately."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from zoom_copilot.translation import TranslationWorker


def finish(worker):
    worker.stop()
    worker.thread.join(3)
    assert not worker.thread.is_alive()


def test_final_overtakes_pending_preview_and_invalidates_running_old_preview():
    running, release, final_done, preview_done = (threading.Event() for _ in range(4))
    calls, results, previews = [], [], []
    def translate(text):
        calls.append(text)
        if text == "old preview":
            running.set()
            assert release.wait(3)
        return "ZH:" + text
    def final(identity, zh):
        results.append((identity, zh))
        final_done.set()
    def preview(identity, source, zh):
        previews.append((identity, source, zh))
        preview_done.set()
    worker = TranslationWorker(SimpleNamespace(translate=translate), final,
                               on_preview=preview, preview_interval=0)
    worker.start()
    try:
        worker.submit_preview("a", "old preview")
        assert running.wait(3)
        worker.submit_preview("b", "obsolete pending preview")
        worker.submit_preview("b", "latest pending preview")
        worker.submit("a", "confirmed final")
        assert worker.submit_preview("a", "too late") is False
        release.set()
        assert final_done.wait(3) and preview_done.wait(3)
        assert calls == ["old preview", "confirmed final", "latest pending preview"]
        assert results == [("a", "ZH:confirmed final")]
        assert previews == [("b", "latest pending preview", "ZH:latest pending preview")]
    finally:
        release.set()
        finish(worker)


def test_updated_english_suppresses_inflight_preview_for_same_entry():
    running, release, complete = (threading.Event() for _ in range(3))
    previews = []
    def translate(text):
        if text == "first hypothesis":
            running.set()
            assert release.wait(3)
        return "ZH:" + text
    def result(*args):
        previews.append(args)
        complete.set()
    worker = TranslationWorker(SimpleNamespace(translate=translate), lambda *args: None,
                               on_preview=result, preview_interval=0)
    worker.start()
    try:
        worker.submit_preview("entry", "first hypothesis")
        assert running.wait(3)
        worker.submit_preview("entry", "revised hypothesis")
        release.set()
        assert complete.wait(3)
        assert previews == [("entry", "revised hypothesis", "ZH:revised hypothesis")]
    finally:
        release.set()
        finish(worker)


def test_preview_throttle_never_delays_ready_final_job():
    first_done, final_done, second_done = (threading.Event() for _ in range(3))
    calls, previews = [], []
    def translate(text):
        calls.append((text, time.monotonic()))
        return text
    def preview(identity, source, zh):
        previews.append(source)
        (first_done if source == "first" else second_done).set()
    worker = TranslationWorker(SimpleNamespace(translate=translate), lambda *args: final_done.set(),
                               on_preview=preview, preview_interval=.35)
    worker.start()
    try:
        worker.submit_preview("a", "first")
        assert first_done.wait(3)
        worker.submit_preview("b", "obsolete")
        worker.submit_preview("b", "latest")
        worker.submit("c", "final")
        assert final_done.wait(.25), "A final must wake a throttled preview wait immediately"
        assert second_done.wait(3)
        assert [text for text, _ in calls] == ["first", "final", "latest"]
        assert calls[2][1] - calls[0][1] >= .3
        assert previews == ["first", "latest"]
    finally:
        finish(worker)


def test_stop_discards_inflight_preview():
    running, release = threading.Event(), threading.Event()
    previews = []
    def translate(text):
        running.set()
        assert release.wait(3)
        return text
    worker = TranslationWorker(SimpleNamespace(translate=translate), lambda *args: None,
                               on_preview=lambda *args: previews.append(args))
    worker.start()
    worker.submit_preview("entry", "pending")
    assert running.wait(3)
    worker.stop()
    release.set()
    finish(worker)
    assert previews == []
