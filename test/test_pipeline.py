"""
Unit tests for Review-1 pipeline components.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import json
import tempfile
import os
from pathlib import Path
from PIL import Image

# ── import pipeline components ──────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import (
    majority_vote,
    ChromaVectorDB,
    evaluate,
    LABEL_NAMES,
    load_dataset,
)


# ─────────────────────────────────────────────
# majority_vote tests
# ─────────────────────────────────────────────
class TestMajorityVote:
    def test_all_ai(self):
        pred, expl = majority_vote([1, 1, 1, 1, 1])
        assert pred == 1
        assert "AI-Generated" in expl

    def test_all_real(self):
        pred, expl = majority_vote([0, 0, 0, 0, 0])
        assert pred == 0
        assert "Real" in expl

    def test_majority_ai(self):
        pred, expl = majority_vote([1, 1, 1, 0, 0])
        assert pred == 1

    def test_majority_real(self):
        pred, expl = majority_vote([0, 0, 0, 1, 1])
        assert pred == 0

    def test_tie_goes_ai(self):
        # k=4 tie: >= → AI
        pred, _ = majority_vote([1, 1, 0, 0])
        assert pred == 1

    def test_explanation_contains_counts(self):
        # 3 AI, 7 Real → predicts Real, explanation should mention 7 (winning count)
        _, expl = majority_vote([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        assert "7 out of 10" in expl
        assert "Real" in expl

    def test_single_element_ai(self):
        pred, _ = majority_vote([1])
        assert pred == 1

    def test_single_element_real(self):
        pred, _ = majority_vote([0])
        assert pred == 0


# ─────────────────────────────────────────────
# ChromaVectorDB tests
# ─────────────────────────────────────────────
class TestChromaVectorDB:
    DIM = 768

    def _make_fake_paths(self, n: int, tmpdir: str) -> list:
        """Create n dummy image files and return their paths."""
        paths = []
        for i in range(n):
            subdir = "ai" if i % 2 == 0 else "real"
            d = Path(tmpdir) / subdir
            d.mkdir(exist_ok=True)
            p = d / f"img_{i:04d}.png"
            Image.new("RGB", (8, 8)).save(p)
            paths.append(p)
        return paths

    def _make_db_with_data(self, tmpdir: str):
        db = ChromaVectorDB(persist_dir=tmpdir)
        rng = np.random.default_rng(0)
        embeddings = rng.random((20, self.DIM)).astype("float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        labels = [i % 2 for i in range(20)]
        paths = self._make_fake_paths(20, tmpdir)
        db.add(embeddings, labels, paths)
        return db, embeddings, labels

    def test_collection_count_after_add(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db, _, _ = self._make_db_with_data(tmpdir)
            assert db.collection.count() == 20

    def test_is_populated_true_after_add(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db, _, _ = self._make_db_with_data(tmpdir)
            assert db.is_populated() is True

    def test_is_populated_false_on_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ChromaVectorDB(persist_dir=tmpdir)
            assert db.is_populated() is False

    def test_search_returns_k_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db, embeddings, _ = self._make_db_with_data(tmpdir)
            _, ret_labels, _ = db.search(embeddings[:3], k=5)
            assert len(ret_labels) == 3
            for row in ret_labels:
                assert len(row) == 5

    def test_search_labels_are_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db, embeddings, _ = self._make_db_with_data(tmpdir)
            _, ret_labels, _ = db.search(embeddings[:3], k=5)
            for row in ret_labels:
                for lbl in row:
                    assert lbl in (0, 1)

    def test_metadata_keys_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db, embeddings, _ = self._make_db_with_data(tmpdir)
            _, _, ret_meta = db.search(embeddings[:1], k=3)
            for meta in ret_meta[0]:
                for key in ("label", "label_name", "filename", "class_dir", "resolution"):
                    assert key in meta

    def test_clear_resets_collection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db, _, _ = self._make_db_with_data(tmpdir)
            assert db.is_populated()
            db.clear()
            assert not db.is_populated()


# ─────────────────────────────────────────────
# evaluate tests
# ─────────────────────────────────────────────
class TestEvaluate:
    def test_perfect_accuracy(self):
        y = [0, 0, 1, 1, 0, 1]
        metrics = evaluate(y, y)
        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_zero_accuracy(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        metrics = evaluate(y_true, y_pred)
        assert metrics["accuracy"] == pytest.approx(0.0)

    def test_balanced_metrics(self):
        y_true = [1, 1, 1, 1, 0, 0, 0, 0]
        y_pred = [1, 1, 0, 0, 0, 0, 1, 1]
        metrics = evaluate(y_true, y_pred)
        assert metrics["accuracy"]  == pytest.approx(0.5)
        assert metrics["precision"] == pytest.approx(0.5)
        assert metrics["recall"]    == pytest.approx(0.5)
        assert metrics["f1"]        == pytest.approx(0.5)

    def test_confusion_matrix_shape(self):
        y = [0, 0, 1, 1]
        metrics = evaluate(y, y)
        cm = np.array(metrics["confusion_matrix"])
        assert cm.shape == (2, 2)

    def test_metrics_dict_keys(self):
        y = [0, 1, 0, 1]
        metrics = evaluate(y, y)
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in metrics

    def test_all_pred_one_class(self):
        y_true = [0, 1, 0, 1]
        y_pred = [1, 1, 1, 1]
        metrics = evaluate(y_true, y_pred)
        assert 0.0 <= metrics["accuracy"] <= 1.0


# ─────────────────────────────────────────────
# load_dataset tests
# ─────────────────────────────────────────────
class TestLoadDataset:
    def _create_fake_dataset(self, tmpdir: str, n_ai=5, n_real=5) -> Path:
        base = Path(tmpdir) / "dataset"
        for cls in ("ai", "real"):
            (base / cls).mkdir(parents=True)
        n = n_ai
        for cls, count in [("ai", n_ai), ("real", n_real)]:
            for i in range(count):
                img = Image.new("RGB", (64, 64), color=(i * 10, i * 20, i * 30))
                img.save(base / cls / f"img_{i:04d}.png")
        return base

    def test_loads_correct_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = self._create_fake_dataset(tmpdir, n_ai=5, n_real=5)
            paths, labels = load_dataset(base)
        assert len(paths) == 10
        assert labels.count(1) == 5
        assert labels.count(0) == 5

    def test_labels_are_0_and_1(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = self._create_fake_dataset(tmpdir, n_ai=3, n_real=4)
            _, labels = load_dataset(base)
        assert set(labels) == {0, 1}

    def test_missing_class_dir_doesnt_crash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "dataset"
            (base / "ai").mkdir(parents=True)
            img = Image.new("RGB", (32, 32))
            img.save(base / "ai" / "test.png")
            paths, labels = load_dataset(base)
        assert len(paths) == 1

    def test_empty_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "dataset"
            paths, labels = load_dataset(base)
        assert paths == []
        assert labels == []