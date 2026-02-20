"""
Review-1: AI vs Real Art Classification Pipeline
Core end-to-end feasibility pipeline using vision embeddings + Chroma vector DB + majority vote.
"""

import os
import json
import time
import uuid
import numpy as np
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel
import chromadb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_DIR     = Path("dataset")
EMBED_DIR       = Path("embeddings")
CHROMA_DIR      = Path("chroma_db")        # Chroma persists here on disk
RESULTS_DIR     = Path("results")
IMG_SIZE        = 224
BATCH_SIZE      = 32
TOP_K           = 10
TEST_SPLIT      = 0.2
RANDOM_SEED     = 42
MODEL_NAME      = "google/vit-base-patch16-224-in21k"
LABEL_MAP       = {"ai": 1, "real": 0}
LABEL_NAMES     = {0: "Real", 1: "AI-Generated"}
COLLECTION_NAME = "art_embeddings"

EMBED_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def load_dataset(dataset_dir: Path) -> Tuple[list, list]:
    """Load all image paths and labels from dataset/ai and dataset/real folders."""
    paths, labels = [], []
    for class_name, label in LABEL_MAP.items():
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            print(f"  [WARNING] Directory not found: {class_dir}")
            continue
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        found = [p for p in class_dir.iterdir() if p.suffix.lower() in exts]
        paths.extend(found)
        labels.extend([label] * len(found))
        print(f"  Loaded {len(found)} images from '{class_name}' class")
    return paths, labels


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
class ViTEmbedder:
    """Wraps a pretrained ViT to extract CLS-token embeddings (frozen)."""

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"\n[Embedder] Loading pretrained ViT: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Embedder] Using device: {self.device}")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.transform = get_transform()

    @torch.no_grad()
    def embed_paths(self, image_paths: list, batch_size: int = BATCH_SIZE) -> np.ndarray:
        all_embeddings = []
        total = len(image_paths)
        for i in range(0, total, batch_size):
            batch_paths = image_paths[i: i + batch_size]
            pixel_values_list = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    tensor = self.transform(img)
                    pixel_values_list.append(tensor)
                except Exception as e:
                    print(f"  [WARN] Could not load {p}: {e}")
                    pixel_values_list.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))

            batch_tensor = torch.stack(pixel_values_list).to(self.device)
            outputs = self.model(pixel_values=batch_tensor)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            cls_embeddings = F.normalize(cls_embeddings, p=2, dim=1)
            all_embeddings.append(cls_embeddings.cpu().numpy())

            pct = min(i + batch_size, total)
            print(f"  Embedded {pct}/{total} images", end="\r")

        print()
        return np.vstack(all_embeddings).astype("float32")


# ─────────────────────────────────────────────
# VECTOR DATABASE (CHROMA)
# ─────────────────────────────────────────────
class ChromaVectorDB:
    """
    Chroma-backed vector store — persists to disk automatically.

    Each stored entry contains:
      - embedding  : 768-d ViT CLS vector
      - metadata   : label (int), label_name, filename, class_dir, resolution
      - document   : filename string (human-readable, used by Chroma internally)
    """

    def __init__(self, persist_dir: str = str(CHROMA_DIR)):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}   # cosine similarity
        )
        print(f"[Chroma] Collection '{COLLECTION_NAME}' loaded "
              f"({self.collection.count()} existing vectors)")

    def is_populated(self) -> bool:
        return self.collection.count() > 0

    def clear(self):
        """Drop and recreate collection for a clean re-index."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print("[Chroma] Collection cleared and recreated.")

    def add(self, embeddings: np.ndarray, labels: list, image_paths: list,
            batch_size: int = 500):
        """
        Insert embeddings with rich metadata into Chroma.
        Metadata stored per vector:
          - label      : 0 or 1
          - label_name : 'Real' or 'AI-Generated'
          - filename   : original image filename
          - class_dir  : 'ai' or 'real' (source folder)
          - resolution : '224x224'
        """
        total = len(embeddings)
        assert total == len(labels) == len(image_paths)

        for i in range(0, total, batch_size):
            batch_emb   = embeddings[i: i + batch_size]
            batch_lbl   = labels[i: i + batch_size]
            batch_paths = image_paths[i: i + batch_size]

            ids = [str(uuid.uuid4()) for _ in batch_emb]

            metadatas = [
                {
                    "label"     : int(lbl),
                    "label_name": LABEL_NAMES[int(lbl)],
                    "filename"  : Path(p).name,
                    "class_dir" : Path(p).parent.name,
                    "resolution": f"{IMG_SIZE}x{IMG_SIZE}",
                }
                for lbl, p in zip(batch_lbl, batch_paths)
            ]

            documents = [Path(p).name for p in batch_paths]

            self.collection.add(
                ids        = ids,
                embeddings = batch_emb.tolist(),
                metadatas  = metadatas,
                documents  = documents,
            )
            print(f"  Indexed {min(i + batch_size, total)}/{total} vectors", end="\r")

        print()
        print(f"[Chroma] Total vectors stored in DB: {self.collection.count()}")

    def search(self, query_embeddings: np.ndarray, k: int = TOP_K):
        """
        Query the collection for each embedding in the batch.
        Returns:
          - distances         : similarity scores
          - retrieved_labels  : list of label lists (one per query)
          - retrieved_metadata: list of metadata dicts (for explainability)
        """
        results = self.collection.query(
            query_embeddings = query_embeddings.tolist(),
            n_results        = k,
            include          = ["metadatas", "distances", "documents"]
        )

        retrieved_labels   = []
        retrieved_metadata = []

        for meta_row in results["metadatas"]:
            retrieved_labels.append([m["label"] for m in meta_row])
            retrieved_metadata.append(meta_row)

        return results["distances"], retrieved_labels, retrieved_metadata


# ─────────────────────────────────────────────
# MAJORITY VOTE CLASSIFIER
# ─────────────────────────────────────────────
def majority_vote(retrieved_labels: list) -> Tuple[int, str]:
    """
    Given a list of neighbour labels (0/1), return
    (predicted_label, explanation_string).
    """
    ai_count   = sum(retrieved_labels)
    real_count = len(retrieved_labels) - ai_count
    predicted  = 1 if ai_count >= real_count else 0
    label_name = LABEL_NAMES[predicted]
    win_count  = ai_count if predicted == 1 else real_count
    explanation = (
        f"This image is classified as {label_name} because "
        f"{win_count} out of {len(retrieved_labels)} "
        f"nearest neighbours are {label_name} images."
    )
    return predicted, explanation


# ─────────────────────────────────────────────
# EVALUATION & REPORTING
# ─────────────────────────────────────────────
def evaluate(y_true: list, y_pred: list) -> dict:
    acc    = accuracy_score(y_true, y_pred)
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=["Real", "AI-Generated"],
        zero_division=0
    )
    metrics = dict(accuracy=acc, precision=prec, recall=rec, f1=f1)

    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"  {k.capitalize():<12}: {v:.4f}")
    print("\n  Classification Report:")
    print(report)

    return dict(**metrics, confusion_matrix=cm.tolist(), report=report)


def plot_confusion_matrix(cm: list, save_path: Path):
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Real", "AI-Generated"],
        yticklabels=["Real", "AI-Generated"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Review-1")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Confusion matrix saved -> {save_path}")


def save_metrics(metrics: dict, save_path: Path):
    serialisable = {k: v for k, v in metrics.items()
                    if k not in ("confusion_matrix", "report")}
    serialisable["confusion_matrix"] = metrics["confusion_matrix"]
    with open(save_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"[Save] Metrics JSON -> {save_path}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline():
    t0 = time.time()
    print("\n" + "=" * 55)
    print("  AI vs Real Art - Review-1 Pipeline (Chroma)")
    print("=" * 55)

    # ── 1. Load dataset ──────────────────────
    print("\n[Step 1] Loading dataset...")
    paths, labels = load_dataset(DATASET_DIR)
    if len(paths) == 0:
        print("[ERROR] No images found. Check dataset/ directory structure.")
        return
    print(f"  Total images: {len(paths)} | AI: {labels.count(1)} | Real: {labels.count(0)}")

    # ── 2. Train / test split ────────────────
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels,
        test_size=TEST_SPLIT,
        stratify=labels,
        random_state=RANDOM_SEED
    )
    print(f"\n[Step 2] Split -> Train: {len(train_paths)} | Test: {len(test_paths)}")

    # ── 3. Feature extraction ─────────────────
    embedder = ViTEmbedder()
    embed_train_file = EMBED_DIR / "train_embeddings.npy"
    embed_test_file  = EMBED_DIR / "test_embeddings.npy"

    if embed_train_file.exists() and embed_test_file.exists():
        print("\n[Step 3] Loading cached embeddings...")
        train_embeddings = np.load(embed_train_file)
        test_embeddings  = np.load(embed_test_file)
    else:
        print("\n[Step 3] Generating train embeddings...")
        train_embeddings = embedder.embed_paths(train_paths)
        print("\n[Step 3] Generating test embeddings...")
        test_embeddings  = embedder.embed_paths(test_paths)
        np.save(embed_train_file, train_embeddings)
        np.save(embed_test_file,  test_embeddings)
        np.save(EMBED_DIR / "train_labels.npy", np.array(train_labels))
        np.save(EMBED_DIR / "test_labels.npy",  np.array(test_labels))
        print(f"  Embeddings cached to {EMBED_DIR}/")

    dim = train_embeddings.shape[1]
    print(f"  Embedding dimension: {dim}")

    # ── 4. Build Chroma vector database ───────
    print("\n[Step 4] Setting up Chroma vector database...")
    db = ChromaVectorDB()

    if db.is_populated():
        print("[Chroma] DB already populated — skipping re-index.")
        print("         (Delete the chroma_db/ folder to force a fresh index)")
    else:
        db.add(train_embeddings, train_labels, train_paths)

    # ── 5. Retrieval + majority vote ──────────
    print(f"\n[Step 5] Classifying test set with Top-K={TOP_K} retrieval...")
    _, retrieved_labels_batch, retrieved_meta_batch = db.search(test_embeddings, k=TOP_K)

    predictions, explanations = [], []
    for nb_labels in retrieved_labels_batch:
        pred, expl = majority_vote(nb_labels)
        predictions.append(pred)
        explanations.append(expl)

    # ── 6. Evaluation ─────────────────────────
    print("\n[Step 6] Evaluating...")
    metrics = evaluate(test_labels, predictions)

    # ── 7. Save artefacts ─────────────────────
    plot_confusion_matrix(metrics["confusion_matrix"], RESULTS_DIR / "confusion_matrix.png")
    save_metrics(metrics, RESULTS_DIR / "metrics.json")

    # Sample explanations with full Chroma metadata
    sample_file = RESULTS_DIR / "sample_explanations.txt"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write("SAMPLE PREDICTIONS & EXPLANATIONS (first 10 test images)\n")
        f.write("=" * 60 + "\n\n")
        for i in range(min(10, len(test_paths))):
            true_lbl = LABEL_NAMES[test_labels[i]]
            pred_lbl = LABEL_NAMES[predictions[i]]
            correct  = "[CORRECT]" if test_labels[i] == predictions[i] else "[WRONG]"
            f.write(f"Image    : {test_paths[i].name}\n")
            f.write(f"True     : {true_lbl}\n")
            f.write(f"Pred     : {pred_lbl}  {correct}\n")
            f.write(f"Reason   : {explanations[i]}\n")
            f.write("Neighbours from Chroma DB:\n")
            for j, meta in enumerate(retrieved_meta_batch[i], 1):
                f.write(f"  [{j:02d}] {meta['filename']:<35} "
                        f"class={meta['class_dir']:<5}  "
                        f"label={meta['label_name']}\n")
            f.write("-" * 60 + "\n")
    print(f"[Save] Sample explanations -> {sample_file}")

    elapsed = time.time() - t0
    print(f"\n[Done] Pipeline completed in {elapsed:.1f}s")
    print(f"       Chroma DB persisted at -> {CHROMA_DIR}/")
    print(f"       Results saved to       -> {RESULTS_DIR}/\n")


if __name__ == "__main__":
    run_pipeline()