"""
train_evaluate.py
------------------
End-to-end pipeline for the Hand Gesture Recognition project.

Pipeline stages:
    1. Load the dataset (train.py -> data_utils.load_raw)
    2. Extract HOG features from every image
    3. Train three candidate models (Random Forest, SVM-RBF, MLP neural net)
    4. Evaluate every model on the held-out test set
       (accuracy, precision, recall, F1, confusion matrix)
    5. Save the best model + all evaluation artefacts to disk

Run with:  python3 train_evaluate.py
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from data_utils import CLASS_NAMES, extract_hog_features, load_raw

HDF5_PATH = Path(__file__).resolve().parent.parent / "data" / "gestures_64px.hdf5"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("STAGE 1: Loading dataset")
    print("=" * 60)
    X_train_img, y_train, X_test_img, y_test = load_raw(HDF5_PATH)
    print(f"Train images: {X_train_img.shape}  |  Test images: {X_test_img.shape}")
    print(f"Classes: {CLASS_NAMES}")

    print("\n" + "=" * 60)
    print("STAGE 2: Extracting HOG features")
    print("=" * 60)
    t0 = time.time()
    X_train = extract_hog_features(X_train_img)
    X_test = extract_hog_features(X_test_img)
    print(f"Feature extraction done in {time.time()-t0:.1f}s")
    print(f"Feature vector size: {X_train.shape[1]}")

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        ),
        "SVM_RBF": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
        "MLP_NeuralNet": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=60,
            random_state=42,
            early_stopping=True,
        ),
    }

    results = {}
    print("\n" + "=" * 60)
    print("STAGE 3 & 4: Training + evaluating each model")
    print("=" * 60)
    for name, clf in models.items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy": acc,
            "train_time_sec": train_time,
            "report": report,
            "confusion_matrix": cm.tolist(),
        }
        print(f"\n[{name}] trained in {train_time:.1f}s  ->  test accuracy = {acc*100:.2f}%")
        print(
            classification_report(
                y_test, y_pred, target_names=CLASS_NAMES, zero_division=0
            )
        )

        # Save confusion matrix figure for this model
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_title(f"Confusion Matrix — {name} (acc={acc*100:.1f}%)")
        plt.tight_layout()
        fig.savefig(OUT_DIR / f"confusion_matrix_{name}.png", dpi=150)
        plt.close(fig)

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = models[best_name]
    print("\n" + "=" * 60)
    print(f"STAGE 5: Best model = {best_name} "
          f"(test accuracy {results[best_name]['accuracy']*100:.2f}%). Saving artefacts.")
    print("=" * 60)

    joblib.dump(best_model, OUT_DIR / "best_gesture_model.joblib")

    with open(OUT_DIR / "results_summary.json", "w") as f:
        json.dump(
            {
                "dataset": {
                    "source": "https://github.com/stes/gestures (CC BY 4.0)",
                    "n_train": int(X_train_img.shape[0]),
                    "n_test": int(X_test_img.shape[0]),
                    "image_size": "64x64",
                    "classes": CLASS_NAMES,
                },
                "best_model": best_name,
                "results": results,
            },
            f,
            indent=2,
        )

    # Bar chart comparing model accuracies
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(results.keys())
    accs = [results[n]["accuracy"] * 100 for n in names]
    bars = ax.bar(names, accs, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Model Comparison on Held-out Test Set")
    ax.set_ylim(0, 100)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 1, f"{a:.1f}%", ha="center")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    print("\nAll done. Artefacts written to:", OUT_DIR)


if __name__ == "__main__":
    main()
