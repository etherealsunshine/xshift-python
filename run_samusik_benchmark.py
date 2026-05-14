from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from xshift import XShift


DATA = Path("data/samusik/samusik01_type_markers.csv")


def main() -> int:
    df = pd.read_csv(DATA)
    marker_cols = [c for c in df.columns if c != "population_id"]

    # Reproducible stratified subset to keep exact-neighbor X-shift tractable.
    sample = (
        df.groupby("population_id", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), max(1, round(10000 * len(x) / len(df)))), random_state=7))
        .sample(frac=1.0, random_state=7)
        .reset_index(drop=True)
    )
    if len(sample) > 10000:
        sample = sample.sample(n=10000, random_state=7).reset_index(drop=True)

    X = sample[marker_cols].to_numpy(dtype=float)
    truth = sample["population_id"].astype(str).to_numpy()

    runs = [
        ("angular_k20_no_final_merges", XShift(k=20, metric="angular", root_merge=False, merge_mahalanobis=None)),
        ("angular_k20_java_like", XShift(k=20, metric="angular", root_merge=True, merge_mahalanobis=2.0)),
        ("euclidean_k20_no_final_merges", XShift(k=20, metric="euclidean", root_merge=False, merge_mahalanobis=None)),
    ]

    rows = []
    out = pd.DataFrame({"population_id": truth})
    for name, model in runs:
        labels = model.fit_predict(X)
        out[name] = labels
        rows.append(
            {
                "run": name,
                "n_cells": len(sample),
                "n_markers": len(marker_cols),
                "n_manual_populations": int(pd.Series(truth).nunique()),
                "n_clusters": model.result_.n_clusters,
                "ari_vs_manual": adjusted_rand_score(truth, labels),
                "nmi_vs_manual": normalized_mutual_info_score(truth, labels),
                "min_cluster_size": int(model.result_.cluster_sizes.min()),
                "median_cluster_size": float(np.median(model.result_.cluster_sizes)),
                "max_cluster_size": int(model.result_.cluster_sizes.max()),
            }
        )

    out_dir = Path("data/samusik/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_dir / "samusik01_10k_input.csv", index=False)
    out.to_csv(out_dir / "samusik01_10k_labels.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "samusik01_10k_metrics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
