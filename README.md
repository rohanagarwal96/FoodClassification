# Food Image Classification with EfficientNetB3

A deep learning project that trains a neural network to recognise **61 different types of food** from photographs. Built as a self-contained Jupyter notebook — run every cell top-to-bottom and it handles everything: downloading the data, training the model, and evaluating results.

---

## What This Project Does

Given an image of food, the model predicts what type of food it is (e.g. pizza, sushi, dumplings). It uses a technique called **transfer learning** — instead of training from scratch, it starts from a model already trained on millions of images (ImageNet) and adapts it to recognise food specifically. This means good results even with limited training time.

**Dataset:** [`bjoernjostein/food-classification`](https://www.kaggle.com/datasets/bjoernjostein/food-classification) on Kaggle — 61 food categories with thousands of labelled images.

---

## Before You Start

### 1. Python Environment
You need Python 3.10 or newer. The first cell of the notebook installs all required packages automatically.

### 2. Kaggle API Credentials
The notebook downloads the dataset directly from Kaggle. You need a free Kaggle account and an API token:

1. Log in at [kaggle.com](https://www.kaggle.com) and go to **Settings → API → Create New Token**
2. This downloads a file called `kaggle.json` that looks like:
   ```json
   {"username": "your_username", "key": "your_api_key"}
   ```
3. Place this file at:
   - **Windows:** `C:\Users\<your-name>\.kaggle\kaggle.json`
   - **Mac/Linux:** `~/.kaggle/kaggle.json`

### 3. Running on Google Colab (Recommended)
Training on a laptop CPU will take several hours. **Google Colab** gives you a free GPU and cuts training time to ~30–60 minutes:

1. Open [colab.research.google.com](https://colab.research.google.com) and upload the notebook
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Add a cell at the very top with your Kaggle credentials:
   ```python
   import os
   os.makedirs("/root/.kaggle", exist_ok=True)
   with open("/root/.kaggle/kaggle.json", "w") as f:
       f.write('{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}')
   os.chmod("/root/.kaggle/kaggle.json", 0o600)
   ```

---

## How to Run

Open `food_classification.ipynb` and run every cell from top to bottom. Each section builds on the previous one — **do not skip cells**.

---

## Notebook Sections Explained

### Section 1 — Setup & Imports
*"Get everything ready before we start."*

Installs the required Python packages (PyTorch, timm, scikit-learn, etc.) and imports them. Also sets a **random seed** so results are reproducible — running the notebook twice gives the same outcome. Detects whether a GPU is available and adjusts training length accordingly.

**Key config you can change:**
| Variable | Default | What it controls |
|---|---|---|
| `EPOCHS` | 20 (GPU) / 5 (CPU) | How many times the model sees the training data |
| `BATCH_SIZE` | 32 | How many images are processed at once |
| `LR` | 0.0001 | Learning rate for Phase 1 training |
| `FINE_TUNE_LR` | 0.00001 | Learning rate for Phase 2 fine-tuning |

---

### Section 2 — Data Download
*"Get the dataset onto your machine."*

Downloads the dataset from Kaggle automatically using the API credentials you set up. Then explores the folder structure and reads `train_img.csv` to find which image belongs to which food category. Since the dataset doesn't have a pre-made validation or test split, this section carves them out automatically:
- **70%** of images → training
- **15%** → validation (used during training to monitor progress)
- **15%** → test (held out until the final evaluation in Section 8)

---

### Section 3 — Exploratory Data Analysis (EDA)
*"Understand the data before touching the model."*

Five visualisations that reveal what the dataset looks like:

| Sub-section | What it shows |
|---|---|
| **3.1 Class Distribution** | Bar chart of how many images exist per food category. Reveals class imbalance — some foods have far more examples than others. |
| **3.2 Sample Image Grid** | 25 random training images with their labels and dimensions, so you can see what the data actually looks like. |
| **3.3 Dimension Analysis** | Scatter plot of image widths vs heights, coloured by aspect ratio (square / portrait / landscape). Useful for understanding how much distortion resizing causes. |
| **3.4 Pixel Intensity** | Histograms of red, green, and blue pixel values across 200 training images. Also computes the per-channel mean and standard deviation used for normalisation. |
| **3.5 Similarity Heatmap** | A grid showing how visually similar different food classes are to each other (based on colour). Bright spots indicate classes that may be confused by the model. |

All plots are saved to the `figures/` folder.

---

### Section 4 — Preprocessing & Augmentation
*"Prepare images so the model can learn from them."*

Neural networks require images in a very specific format (fixed size, normalised pixel values). This section defines two pipelines:

- **Training pipeline** — applies random flips, rotations, crops, and colour jitter. This *augmentation* artificially increases the variety of training examples so the model generalises better and doesn't just memorise the training set.
- **Evaluation pipeline** — only resizes and normalises. No random changes — we want consistent, reproducible predictions.

Also defines the `FoodDataset` class that feeds images into the model during training.

---

### Section 5 — Model Architecture
*"Choose and configure the neural network."*

Uses **EfficientNetB3**, a convolutional neural network pre-trained on ImageNet (14 million images, 1000 categories). The pre-trained weights give the model a huge head start — it already knows how to detect edges, textures, and shapes.

A custom **classification head** is added on top:
```
Pre-trained EfficientNetB3 features
    → BatchNorm → Dropout(40%) → Linear(1536→512) → ReLU → Dropout(30%) → Linear(512→61)
```

Training happens in two phases (see Section 6). The `unfreeze_top_fraction()` helper used in Phase 2 is also defined here.

---

### Section 6 — Training
*"Teach the model to recognise food."*

Training uses a **two-phase approach** to avoid destroying the pre-trained knowledge:

**Phase 1 — Feature Extraction (frozen backbone)**
The EfficientNetB3 weights are locked (frozen). Only the new classification head is trained. This is fast and safe — the model learns to map existing ImageNet features to food categories without risking overwriting useful knowledge.

**Phase 2 — Fine-Tuning (partial unfreeze)**
The top 30% of the backbone is unfrozen and trained at a much lower learning rate. This lets the model adapt its deeper features specifically to food images.

Both phases use:
- **Adam optimiser** — adapts the learning rate automatically per parameter
- **Class-weighted loss** — penalises mistakes on rare food categories more heavily, addressing the 30x class imbalance found in Section 3.1
- **Early stopping** — stops training automatically if the model stops improving (prevents wasting time)
- **ReduceLROnPlateau** — halves the learning rate when progress stalls
- **CSV Logger** — saves per-epoch metrics to `training_log.csv`

---

### Section 7 — Training Visualisation
*"See how training went."*

Three side-by-side charts showing what happened during training:
1. **Loss curve** — lower is better; should decrease over time
2. **Accuracy curve** — higher is better; should increase over time
3. **Learning rate schedule** — shows when the LR was reduced

A dashed vertical line marks where Phase 2 fine-tuning began.

---

### Section 8 — Evaluation
*"Rigorously test the final model on data it has never seen."*

Loads the best saved model checkpoint and runs it on the held-out test set. Six sub-sections:

| Sub-section | What it measures |
|---|---|
| **8.1 Core Metrics** | Top-1, Top-3, Top-5 accuracy; Macro F1; Cohen's Kappa — a one-page summary of overall performance |
| **8.2 Per-Class Report** | F1-score for every individual food category; highlights the 5 best and 5 worst performing classes |
| **8.3 Confusion Matrix** | 61×61 grid showing which foods get confused with which; a zoomed version highlights the 15 most common mistakes |
| **8.4 ROC / AUC** | Measures how well the model separates each class from all others; macro-averaged AUC close to 1.0 is ideal |
| **8.5 Misclassification Gallery** | 16 images the model got wrong, showing the true label, predicted label, and confidence — useful for understanding failure modes |
| **8.6 Grad-CAM** | Heatmap overlays showing *which part of the image* the model focused on when making a prediction |

---

### Section 9 — Inference Helper
*"Use the trained model on any food image."*

A reusable `predict_food()` function:
```python
results = predict_food("path/to/your/image.jpg", model, CLASS_NAMES, top_k=5)
# Returns: [("sushi", 0.87), ("sashimi", 0.06), ...]
```
Demonstrated on 3 test images with top-5 predictions printed for each.

---

### Section 10 — Summary & Conclusions
A written summary of the approach, final metrics, key findings from EDA, and five concrete ideas for improving accuracy further (test-time augmentation, mixup, label smoothing, larger backbone, class-weighted loss).

---

## Output Files

After running the full notebook, these files will exist in the project directory:

| File | Description |
|---|---|
| `best_food_classifier.pth` | Saved model weights (best validation loss) |
| `training_log.csv` | Epoch-by-epoch metrics for both training phases |
| `figures/class_distribution.png` | Class imbalance bar chart |
| `figures/sample_grid.png` | 5×5 sample image grid |
| `figures/dimension_scatter.png` | Image size scatter plot |
| `figures/channel_histograms.png` | RGB pixel histograms |
| `figures/class_similarity_heatmap.png` | Visual similarity between classes |
| `figures/training_curves.png` | Loss, accuracy, and LR over time |
| `figures/confusion_matrix.png` | Full 61×61 confusion matrix |
| `figures/confusion_matrix_zoomed.png` | Top 15 most-confused class pairs |
| `figures/roc_curves.png` | ROC curves for top/bottom 10 classes |
| `figures/misclassification_gallery.png` | 16 misclassified examples |
| `figures/gradcam_examples.png` | Grad-CAM heatmap overlays |

---

## Tech Stack

| Library | Purpose |
|---|---|
| `torch` + `torchvision` | Neural network framework and image transforms |
| `timm` | Pre-trained EfficientNetB3 model |
| `kaggle` | Dataset download via Kaggle API |
| `scikit-learn` | Metrics, train/val/test splitting |
| `matplotlib` + `seaborn` | All visualisations |
| `numpy` + `pandas` | Numerical operations and data handling |
| `Pillow` | Image loading and resizing |
| `tqdm` | Progress bars |
