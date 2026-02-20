# AI vs Real Art Detector — Review-1

> Core feasibility pipeline: Vision Embeddings → FAISS Retrieval → Majority Vote Classification

---

## Project Structure

```
ai_art_detector/
├── pipeline.py              ← Main end-to-end pipeline
├── requirements.txt
├── README.md
├── dataset/
│   ├── ai/                  ← 1000 AI-generated art images
│   └── real/                ← 1000 real (human-created) art images
├── embeddings/              ← Auto-created; cached .npy files + FAISS index
├── results/                 ← Auto-created; metrics.json, confusion_matrix.png, sample_explanations.txt
├── tests/
│   └── test_pipeline.py     ← Unit tests (pytest)
└── .github/
    └── workflows/
        └── ci_cd.yml        ← GitHub Actions CI/CD pipeline
```

---

## Setup

### 1. Clone & install dependencies

```bash
git clone <your-repo-url>
cd ai_art_detector
pip install -r requirements.txt
```

### 2. Prepare dataset

Organise your images as:

```
dataset/
  ai/       ← AI-generated images (.jpg / .png / .webp)
  real/     ← Real art images     (.jpg / .png / .webp)
```

### 3. Run the pipeline

```bash
python pipeline.py
```

**What happens:**

| Step | Description |
|------|-------------|
| 1 | Load all images from `dataset/ai/` and `dataset/real/` |
| 2 | Split 80% train / 20% test (stratified) |
| 3 | Extract CLS-token embeddings with frozen `ViT-Base-Patch16` |
| 4 | Index all train embeddings in a FAISS flat inner-product index |
| 5 | For each test image retrieve Top-K=10 nearest neighbours |
| 6 | Predict label by majority vote over neighbour labels |
| 7 | Compute Accuracy, Precision, Recall, F1, Confusion Matrix |
| 8 | Save artefacts to `results/` |

---

## Configuration

All key parameters are at the top of `pipeline.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `IMG_SIZE` | 224 | Resize target (px) |
| `BATCH_SIZE` | 32 | Embedding batch size |
| `TOP_K` | 10 | Neighbours for majority vote |
| `TEST_SPLIT` | 0.2 | Fraction held out for testing |
| `MODEL_NAME` | `google/vit-base-patch16-224-in21k` | HuggingFace ViT model |

---

## Output Files

| File | Contents |
|------|----------|
| `results/metrics.json` | Accuracy, Precision, Recall, F1 |
| `results/confusion_matrix.png` | Visual confusion matrix |
| `results/sample_explanations.txt` | Example predictions with neighbour-based reasoning |
| `embeddings/train_embeddings.npy` | Cached train embeddings (skip re-extraction on re-runs) |
| `embeddings/faiss_index.index` | Serialised FAISS index |

### Sample explanation format

```
Image  : painting_0042.jpg
True   : AI-Generated
Pred   : AI-Generated  ✓
Reason : This image is classified as AI-Generated because 8 out of 10
         nearest neighbours are AI-Generated images.
```

---

## Running Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=pipeline
```

Tests cover:
- `majority_vote` edge cases (ties, single elements, explanation text)
- `FAISSVectorDB` (indexing, search, save/load)
- `evaluate` (accuracy, confusion matrix shape, all-one-class predictions)
- `load_dataset` (counts, missing dirs, empty dataset)

---

## CI/CD (GitHub Actions)

### Continuous Integration — triggered on every push

- **Lint** (`flake8`): fails on syntax errors or undefined names
- **Dataset structure check**: warns if `dataset/ai` or `dataset/real` is missing
- **Unit tests + coverage** (`pytest`)
- Builds fail automatically if any test fails

### Continuous Deployment — triggered on merge to `main`

- Packages `pipeline.py` + `requirements.txt` with a `VERSION.txt`
- Uploads deployment artefact (retained 30 days)

### Secret management

The Gemini API key (for future reviews) is stored as a GitHub Actions secret `GEMINI_API_KEY` and injected via environment variables — never hard-coded.

---

## Pipeline Architecture

```
Image
  │
  ▼
ViT-Base-Patch16 (frozen)
  │  CLS-token embedding (768-d, L2-normalised)
  ▼
FAISS IndexFlatIP
  │  Top-K=10 nearest neighbours
  ▼
Majority Vote
  │  label = argmax(AI_count, Real_count)
  ▼
Prediction + Explanation
```

---

## Review Roadmap

| Review | Features |
|--------|----------|
| **Review-1** ✅ | Embeddings + FAISS retrieval + majority vote + evaluation |
| Review-2 | Gemini API RAG reasoning + probability calibration |
| Review-3 | Grad-CAM / attention heatmaps (visual explainability) |
| Review-4 | FastAPI backend + React frontend web app |
