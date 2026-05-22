import json
import os
import pandas as pd
import numpy as np


# build an entity level confusion matrix with entity level precision and recall 
def entityMetricsTable(experiment_name,
                       rule="strict",
                       results_dir="./results/"):

    filename = f"results_{experiment_name}.json"

    filepath = os.path.join(results_dir, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    entity_metrics = data["entity_metrics"][rule]

    # actual labels (rows)
    rows = ["PER", "LOC", "ORG", "NaNE"]

    # predicted labels (columns)
    cols = ["PER", "LOC", "ORG", "O"]

    cm = np.array(entity_metrics["confusion_matrix"])

    precision = entity_metrics["precision"]
    recall = entity_metrics["recall"]
    support = entity_metrics["support"]

    # build dataframe
    df = pd.DataFrame(
        cm,
        index=rows,
        columns=cols
    )

    df[cols] = df[cols].map(lambda x: str(int(x)))

    # NaNE/O does not apply
    df.loc["NaNE", "O"] = "-"

    # add recall column (exclude O)
    df["Recall"] = [
        round(recall["PER"], 3),
        round(recall["LOC"], 3),
        round(recall["ORG"], 3),
        ""
    ]

    df["Support"] = [
        round(support["PER"], 3),
        round(support["LOC"], 3),
        round(support["ORG"], 3),
        "" 
    ]

    # precision row (exclude O)
    precision_row = [
        round(precision["PER"], 3),
        round(precision["LOC"], 3),
        round(precision["ORG"], 3),
        "",
        "",
        ""
    ]

    df.loc["Precision"] = precision_row

    return df




def spanMetricsTable(experiment_name,
                     results_dir="results/"):

    filename = f"results_{experiment_name}.json"

    filepath = os.path.join(results_dir, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    span_metrics = data["span_metrics"]

    rows = [
        ("Exact Match", "exact_match"),
        ("Unlabeled Match", "unlabeled_match"),
        ("Loose Match", "loose_match")
    ]

    table = []

    for display_name, key in rows:

        metric = span_metrics[key]

        table.append({
            "Metric": display_name,
            "Precision": round(metric["precision"], 3),
            "Recall": round(metric["recall"], 3),
            "F1": round(metric["f1"], 3)
        })

    df = pd.DataFrame(table).set_index("Metric")

    return df