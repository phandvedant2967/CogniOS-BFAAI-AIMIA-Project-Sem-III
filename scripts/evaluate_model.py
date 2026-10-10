"""CLI Script to evaluate a trained model checkpoint on any dataset split."""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ml.data import load_dataset, extract_split_data
from ml.models import ModelWrapper
from ml.preprocessing import PreprocessorPipeline
from ml.evaluation import evaluate_predictions
from ml.utils import MODELS_DIR


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained ML model checkpoint")
    parser.add_argument("--model", default=os.path.join(MODELS_DIR, "baseline_random_forest.pkl"), help="Path to model pkl")
    parser.add_argument("--prep", default=os.path.join(MODELS_DIR, "preprocessing.pkl"), help="Path to preprocessor pkl")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split to evaluate")
    args = parser.parse_args()

    print(f"Loading preprocessor from: {args.prep}")
    prep = PreprocessorPipeline.load(args.prep)

    print(f"Loading model from: {args.model}")
    model = ModelWrapper.load(args.model)

    print(f"Loading V2 dataset ({args.split} split)...")
    df = load_dataset()
    X, y, meta, _ = extract_split_data(df, args.split)

    X_proc = prep.transform(X)

    preds = model.predict(X_proc)
    probs = model.predict_proba(X_proc)

    metrics = evaluate_predictions(y, preds, probs, model_name=f"{model.model_type}_{args.split}")

    print(f"\n=======================================================")
    print(f"EVALUATION RESULTS ({model.model_type} on {args.split.upper()} split)")
    print(f"=======================================================")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:     {metrics['weighted_f1']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.4f}")

    print("\nPer-Class Breakdown:")
    for cls, m in metrics["per_class"].items():
        print(f"  - {cls:<22}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f} (support={m['support']})")


if __name__ == "__main__":
    main()
