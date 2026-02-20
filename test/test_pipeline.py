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
    FAISSVectorDB,
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
# FAISSVectorDB tests
# ─────────────────────────────────────────────
class TestFAISSVectorDB:
    DIM = 768

    def _make_db_with_data(self):
        db = FAISSVectorDB(self.DIM)
        rng = np.random.default_rng(0)
        embeddings = rng.random((100, self.DIM)).astype("float32")
        # L2 normalise (as pipeline does)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        labels = [i % 2 for i in range(100)]   # alternating 0/1
        db.add(embeddings, labels)
        return db, embeddings, labels

    def test_index_count_after_add(self):
        db, _, _ = self._make_db_with_data()
        assert db.index.ntotal == 100

    def test_labels_stored(self):
        db, _, labels = self._make_db_with_data()
        assert len(db.labels) == 100
        assert db.labels == labels

    def test_search_returns_k_results(self):
        db, embeddings, _ = self._make_db_with_data()
        query = embeddings[:5]
        _, _, ret_labels = db.search(query, k=10)
        assert len(ret_labels) == 5
        for row in ret_labels:
            assert len(row) == 10

    def test_search_labels_are_valid(self):
        db, embeddings, _ = self._make_db_with_data()
        _, _, ret_labels = db.search(embeddings[:3], k=5)
        for row in ret_labels:
            for lbl in row:
                assert lbl in (0, 1)

    def test_nearest_neighbour_is_self(self):
        db, embeddings, _ = self._make_db_with_data()
        query = embeddings[:1]
        distances, indices, _ = db.search(query, k=1)
        assert indices[0][0] == 0    # self is closest

    def test_save_and_load(self):
        db, embeddings, labels = self._make_db_with_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_index")
            db.save(path)
            db2 = FAISSVectorDB.load(path, self.DIM)
        assert db2.index.ntotal == 100
        assert db2.labels == labels


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