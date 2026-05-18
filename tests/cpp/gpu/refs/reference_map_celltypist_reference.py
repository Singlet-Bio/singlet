#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/tests/refs/reference_map_celltypist_reference.py
#
# Reference driver: train a small CellTypist-style logistic regression model on
# a labeled synthetic embedding, then project a held-out test set.
# Saves model coefficients to a .npz (the C++ 'load_celltypist_model' fixture)
# and ground-truth predictions to a separate .npz.
#
# Environment: pip install celltypist scanpy scikit-learn numpy scipy
#
# Usage (two modes):
#
#   # Mode 1: generate both model and predictions from a synthetic embedding
#   python reference_map_celltypist_reference.py \
#       --mode       generate \
#       --train-emb  <train.npy>       # float32 (n_train × n_pcs)
#       --train-lab  <train_labels.npy># int32   (n_train,)
#       --test-emb   <test.npy>        # float32 (n_test  × n_pcs)
#       --class-names <names.npy>      # str array (n_classes,) — class name per label id
#       --model-out  <model.npz>       # C++ fixture: weights, intercepts, class_names
#       --pred-out   <predictions.npz> # ground-truth predictions for the test set
#
#   # Mode 2: just project a test set using an existing model .npz
#   python reference_map_celltypist_reference.py \
#       --mode      project \
#       --model-in  <model.npz> \
#       --test-emb  <test.npy>  \
#       --pred-out  <predictions.npz>
#
# model.npz keys (C++ load_celltypist_model format):
#   weights      : float32 (n_classes × n_pcs)
#   intercepts   : float32 (n_classes,)
#   class_names  : str array (n_classes,)
#
# predictions.npz keys:
#   labels       : int32  (n_cells,)   — argmax class index
#   probabilities: float32(n_cells × n_classes) — softmax probabilities
#   class_names  : str array (n_classes,)

import argparse
import sys
import numpy as np
from scipy.special import softmax


def train_logistic_model(train_emb: np.ndarray,
                         train_lab: np.ndarray,
                         class_names: list[str]) -> dict:
    """
    Fit sklearn LogisticRegression (same objective as CellTypist) and extract
    coefficients in the C++ load_celltypist_model .npz format.
    """
    from sklearn.linear_model import LogisticRegression

    n_classes = len(class_names)
    print(f"[ref] Training LR: {train_emb.shape[0]} cells, "
          f"{train_emb.shape[1]} PCs, {n_classes} classes", file=sys.stderr)

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        random_state=42,
    )
    clf.fit(train_emb, train_lab)

    # coef_ shape: (n_classes, n_features) — exactly what we need.
    weights    = clf.coef_.astype(np.float32)     # (n_classes, n_pcs)
    intercepts = clf.intercept_.astype(np.float32) # (n_classes,)

    print(f"[ref] weights: {weights.shape}, intercepts: {intercepts.shape}",
          file=sys.stderr)
    return {
        "weights":     weights,
        "intercepts":  intercepts,
        "class_names": np.array(class_names, dtype=object),
    }


def project_model(model: dict,
                  test_emb: np.ndarray) -> dict:
    """Apply the logistic model to a test embedding, return labels + probs."""
    weights    = model["weights"]      # (n_classes, n_pcs)
    intercepts = model["intercepts"]   # (n_classes,)
    class_names = model["class_names"]

    # logits: (n_cells, n_classes)
    logits = test_emb @ weights.T + intercepts[np.newaxis, :]
    probs  = softmax(logits, axis=1).astype(np.float32)
    labels = np.argmax(probs, axis=1).astype(np.int32)

    print(f"[ref] projected {test_emb.shape[0]} cells → "
          f"{len(np.unique(labels))} unique classes", file=sys.stderr)
    return {
        "labels":        labels,
        "probabilities": probs,
        "class_names":   class_names,
    }


def main():
    parser = argparse.ArgumentParser(
        description="CellTypist reference driver for singlet-gpu anno/reference_map tests")
    parser.add_argument("--mode", required=True, choices=["generate", "project"])
    parser.add_argument("--train-emb",  help="float32 .npy (n_train × n_pcs)")
    parser.add_argument("--train-lab",  help="int32  .npy (n_train,) — 0-indexed class labels")
    parser.add_argument("--test-emb",   required=True,
                        help="float32 .npy (n_test × n_pcs)")
    parser.add_argument("--class-names",
                        help="str .npy (n_classes,) — class name for each label id")
    parser.add_argument("--model-in",  help="Existing model .npz (project mode)")
    parser.add_argument("--model-out", help="Output model .npz (generate mode)")
    parser.add_argument("--pred-out",  required=True, help="Output predictions .npz")
    args = parser.parse_args()

    test_emb = np.load(args.test_emb).astype(np.float32)
    print(f"[ref] test embedding: {test_emb.shape}", file=sys.stderr)

    if args.mode == "generate":
        if not args.train_emb or not args.train_lab:
            parser.error("--mode generate requires --train-emb and --train-lab")
        train_emb = np.load(args.train_emb).astype(np.float32)
        train_lab = np.load(args.train_lab).astype(np.int32)

        if args.class_names:
            raw = np.load(args.class_names, allow_pickle=True)
            class_names = [str(x) for x in raw]
        else:
            n_cls = int(train_lab.max()) + 1
            class_names = [f"class_{i}" for i in range(n_cls)]

        model = train_logistic_model(train_emb, train_lab, class_names)

        if args.model_out:
            np.savez(args.model_out, **model)
            print(f"[ref] Saved model to {args.model_out}", file=sys.stderr)

    elif args.mode == "project":
        if not args.model_in:
            parser.error("--mode project requires --model-in")
        raw = np.load(args.model_in, allow_pickle=True)
        model = {
            "weights":     raw["weights"].astype(np.float32),
            "intercepts":  raw["intercepts"].astype(np.float32),
            "class_names": raw["class_names"],
        }
        print(f"[ref] Loaded model: {model['weights'].shape[0]} classes, "
              f"{model['weights'].shape[1]} PCs", file=sys.stderr)

    preds = project_model(model, test_emb)
    np.savez(args.pred_out, **preds)
    print(f"[ref] Saved predictions to {args.pred_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
