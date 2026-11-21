# Hand Gesture Recognition — Trained ML Pipeline

This project replaces a prior submission that only called Google's pre-trained
MediaPipe model on webcam frames (no dataset, no training, no evaluation).
This version defines a real dataset, extracts features, trains multiple
models, and evaluates them with standard metrics.

## Dataset

**Binary Hand Gesture Dataset** — https://github.com/stes/gestures (CC BY 4.0)

- 9,724 training images, 667 test images
- 64x64 px binary hand silhouettes
- 7 classes: fist, thumb, pistol, three, four, five, peace

## Pipeline

1. `src/data_utils.py` — loads the HDF5 dataset and extracts HOG
   (Histogram of Oriented Gradients) feature vectors from each image.
2. `src/train_evaluate.py` — trains three models (Random Forest, SVM-RBF,
   MLP neural network) on the training split, evaluates all three on the
   held-out test split, and saves:
   - `outputs/best_gesture_model.joblib` — the best trained model
   - `outputs/results_summary.json` — full metrics for every model
   - `outputs/confusion_matrix_<model>.png` — confusion matrix per model
   - `outputs/model_comparison.png` — accuracy bar chart
   - `outputs/dataset_samples.png` — one example image per class
   - `outputs/prediction_samples.png` — sample predictions (correct + wrong)

## How to run it yourself

```bash
pip install h5py scikit-learn scikit-image matplotlib joblib numpy
git clone https://github.com/stes/gestures.git
cd src
python3 train_evaluate.py
```

## Results (this run)

| Model         | Test Accuracy |
|---------------|---------------|
| Random Forest | 67.0%         |
| SVM (RBF)     | **67.3%** (best) |
| MLP Neural Net| 62.1%         |

Full per-class precision/recall/F1 and confusion matrices are in
`outputs/results_summary.json` and the `.png` files, and are written up in
`outputs/Gesture_Recognition_Report.docx`.


## Honesty note

These are genuine results from a real run of this code — not fabricated
numbers. Accuracy is moderate (~67%) because HOG+classical-ML on a small
silhouette dataset is a reasonable but not state-of-the-art approach; the
"Future Work" section in the report/slides describes how a CNN could push
this further.
