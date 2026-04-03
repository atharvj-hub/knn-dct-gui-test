# KNN & DCT – Interactive Learning Tool

An interactive Python application for learning two fundamental algorithms in
machine learning and signal processing:

| Algorithm | Full name | Where is it used? |
|-----------|-----------|-------------------|
| **KNN** | K-Nearest Neighbours | Classification, recommendation systems |
| **DCT** | Discrete Cosine Transform | JPEG image compression, MP3 audio |

---

## What you will learn

### K-Nearest Neighbours (KNN)

* How Euclidean distance is used to find "similar" training samples
* How the parameter **k** controls the complexity of the decision boundary
* Why small k can overfit and large k can underfit

### Discrete Cosine Transform (DCT)

* How a signal can be expressed as a sum of cosine basis functions
* **Energy compaction** – why most signal information lives in low-frequency
  coefficients
* How discarding high-frequency coefficients reconstructs an approximation
  of the original (the basis of JPEG and MP3 compression)

---

## Project layout

```
knn-dct-gui-test/
├── app.py          ← launch this to open the GUI
├── knn.py          ← KNN classifier + data helpers
├── dct.py          ← DCT/IDCT implementation + compression helpers
├── requirements.txt
└── tests/
    ├── test_knn.py
    └── test_dct.py
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the GUI
python app.py
```

The window opens with two tabs:

| Tab | What you can do |
|-----|-----------------|
| **KNN** | Drag the **k** slider and watch the decision boundary reshape in real time |
| **DCT** | Drag the **coefficients kept** slider and observe signal reconstruction quality |

---

## Running the tests

```bash
pytest tests/ -v
```

---

## How the code is structured

### `knn.py`

| Symbol | Purpose |
|--------|---------|
| `euclidean_distance(a, b)` | Returns ‖a − b‖₂ |
| `KNNClassifier(k)` | `fit(X, y)` → `predict(X)` → `score(X, y)` |
| `make_two_class_blobs(...)` | Generates two Gaussian clusters for demo |
| `make_decision_grid(...)` | Dense meshgrid for boundary plotting |

### `dct.py`

| Symbol | Purpose |
|--------|---------|
| `dct1d(x)` / `idct1d(X)` | 1-D DCT-II and its inverse |
| `dct2d(block)` / `idct2d(B)` | 2-D DCT-II and its inverse |
| `compress_1d(x, keep)` | Zero-out high-freq. coefficients, reconstruct |
| `compress_2d(block, keep_rows, keep_cols)` | Same for 2-D blocks |
| `make_demo_signal(N)` | Smooth test signal (sum of sinusoids) |
| `make_demo_image(size)` | Gradient + checkerboard test image |
