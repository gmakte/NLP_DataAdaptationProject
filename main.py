import argparse
import json
import os
from datetime import datetime

# from train import train_model
# from predict import predict
from eval import evaluate_predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)

    #useful for evaluation mode
    parser.add_argument("--pred_path", default="predictions/preds_mixed_2.txt")
    parser.add_argument("--ref_path", default="data/FIN3_fixed.txt")
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--output_file", default="results_mixed_2.json")

    args = parser.parse_args()

    # if args.mode == "train":
    #     train_model(args)

    # elif args.mode == "predict":
    #     predict(args)

    if args.mode == "evaluate":
        results = evaluate_predictions(
            preds=args.pred_path,
            refs=args.ref_path
        )

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, args.output_file)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=4, default=lambda x: float(x))

        print(f"Results saved to {output_path}")

    # elif args.mode == "train_predict":
    #     train_model(args)
    #     predict(args)


if __name__ == "__main__":
    main()