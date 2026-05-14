import time
from pathlib import Path
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from xshift import XShift

df = pd.read_csv("data/samusik/samusik01_type_markers.csv")
X = df.drop(columns=["population_id"]).to_numpy(float)
truth = df["population_id"].astype(str)
t0 = time.time()
model = XShift(
    k=20,
    metric="angular",
    root_merge=True,
    merge_mahalanobis=2.0,
    verbose=True,
    valley_knn_backend="hnsw",
    hnsw_ef=120,
)
labels = model.fit_predict(X)
seconds = time.time() - t0
out_dir = Path("data/samusik/results")
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"population_id": truth, "python_angular_k20_fast_hnsw": labels}).to_csv(out_dir / "python_samusik01_full_fast_hnsw_labels.csv", index=False)
summary = pd.DataFrame([{
    "run": "python_angular_k20_full_fast_hnsw",
    "n_cells": len(labels),
    "n_manual_populations": truth.nunique(),
    "n_clusters": model.result_.n_clusters,
    "ari_vs_manual": adjusted_rand_score(truth, labels),
    "nmi_vs_manual": normalized_mutual_info_score(truth, labels),
    "min_cluster_size": int(pd.Series(labels).value_counts().min()),
    "median_cluster_size": float(pd.Series(labels).value_counts().median()),
    "max_cluster_size": int(pd.Series(labels).value_counts().max()),
    "seconds": seconds,
}])
summary.to_csv(out_dir / "python_samusik01_full_fast_hnsw_metrics.csv", index=False)
print(summary.to_string(index=False))
