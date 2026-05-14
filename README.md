# X-shift Python Port

This is a Python port of the clustering core used by Nolan Lab VorteX's
`Xshift.java` wrapper. The Java wrapper handles FCS/config I/O, then delegates
the actual clustering to `vortex.clustering.XShiftClustering`; this repo ports
that clustering logic and includes a Java-style standalone wrapper for cytometry
files.

## Install

```bash
python3 -m pip install -r requirements.txt
```

Optional FCS I/O:

```bash
python3 -m pip install fcsparser flowio fcswrite
```

## What Is Ported

- K-nearest-neighbor density: density is the negative sum of distances to the
  first `k` neighbors, matching `getDensityWithKNN`.
- Higher-density parent links over a local neighbor list.
- Gradient assignment by following parent links to a surviving root.
- Root filtering/merging using Gabriel-neighbor checks and density along the
  line between roots.
- Final greedy diagonal Mahalanobis merge with the Java default threshold `2.0`.
- Java-style automatic `n_size` selection.
- Optional HNSW nearest-neighbor backend for faster root-valley checks.
- Cached Mahalanobis merge using per-cluster sufficient statistics.
- Experimental Java-style tessellation backend for the main KNN graph
  (`main_knn_backend="tessellation"`).

The Java code uses a tesselation shortcut to accelerate nearest-neighbor
searches. This Python version uses exact scikit-learn for the main KNN graph and
can use `hnswlib` for fast root-valley checks. A vectorized Python port of the
Java tessellation KNN shortcut is also available. It matched sklearn on the
Samusik 10k benchmark, but it is currently experimental and not always faster
than sklearn in Python.

## Python API

```python
import pandas as pd
from xshift import XShift

df = pd.read_csv("events.csv")
markers = ["CD3", "CD4", "CD8", "CD45"]
X = df[markers].to_numpy(float)

model = XShift(
    k=20,
    metric="angular",
    root_merge=True,
    merge_mahalanobis=2.0,
    valley_knn_backend="hnsw",
    # main_knn_backend="tessellation",  # optional experimental Java-style KNN
)
labels = model.fit_predict(X)
df["xshift_label"] = labels
df.to_csv("events_xshift.csv", index=False)

print(model.result_.n_clusters)
print(model.result_.cluster_sizes)
print(model.result_.centers)
```

For raw density modes before the Java root-filtering stage:

```python
labels = XShift(k=20, metric="angular", root_merge=False).fit_predict(X)
```

## CSV CLI

```bash
python3 xshift.py sample.csv --features CD3,CD4,CD8,CD45 --k 20 --output clustered.csv
```

Feature columns can also be 1-based indexes, similar to the original config:

```bash
python3 xshift.py sample.csv --features 3,4,7,8 --k auto --output clustered.csv
```

The output is the input CSV plus an `xshift_label` column.

## Notes For Cytometry

VorteX's standalone wrapper used angular distance by default. That is often
reasonable for marker-profile shape, but Euclidean distance may be preferable
after arcsinh/logicle transformation and per-channel scaling:

```python
labels = XShift(k=30, metric="euclidean").fit_predict(X)
```

The core `XShift` class expects a numeric matrix. For FCS files, use
`xshift_standalone.py` with optional `fcsparser`/`flowio` readers. FCS writing
requires `fcswrite`; otherwise the wrapper writes clustered CSV files.

## Java-Style Standalone Wrapper

`xshift_standalone.py` ports the command-line behavior of
`standalone/Xshift.java`:

```bash
python3 xshift_standalone.py 20
```

By default it reads `importConfig.txt` and `fcsFileList.txt`, writes into `out/`,
and accepts `auto` in place of `20`.

```bash
python3 xshift_standalone.py auto --config importConfig.txt --file-list fcsFileList.txt --out out
```

The wrapper implements the Java `DatasetImporter` preprocessing rules:
`noise_threshold`, `transformation`, `scaling_factor`,
`euclidian_length_threshold`, `limit_events_per_file`, and global `rescale`
with `SD` or `QUANTILE`.

For FCS input, install either `fcsparser` or `flowio` for reading. FCS writing
requires `fcswrite`; otherwise the wrapper writes one clustered CSV per input
file plus `mst.gml`.

## Benchmarks

Validation used the Samusik mouse bone marrow CyTOF dataset from PubMed
27183440 via HDCytoData (`Samusik_01_SE`; 86,864 cells, 39 type markers, 25
manual population labels).

Full dataset, `k=20`:

| Run | Time | Clusters | ARI vs manual | NMI vs manual |
|---|---:|---:|---:|---:|
| Java VorteX | not timed cleanly | 82 | 0.1783 | 0.4363 |
| Python original Java-like | 1008s | 78 | 0.1893 | 0.4446 |
| Python HNSW + cached merge | 82.6s | 78 | 0.1893 | 0.4446 |
| Python tessellation + HNSW + cached merge | 84.8s | 78 | 0.1893 | 0.4446 |

Python tessellation + HNSW + cached merge vs Java labels:

```text
ARI = 0.9490
NMI = 0.9753
```

On this full Samusik run, `main_knn_backend="tessellation"` produced the same
labels as the exact sklearn main KNN path with HNSW/cached merge
(`ARI = 1.0`, `NMI = 1.0` between the two Python outputs). The cached merge row
is therefore the fastest Python run measured here, while the tessellation path
is a Java-style alternative KNN implementation for parity experiments.

The port is therefore behaviorally close to Java, but not bit-identical. The
remaining differences are likely from Java's tesselated nearest-neighbor search
and ordering/tie details.

## Benchmark Scripts

The benchmark scripts are optional and expect the Samusik data to be available
under `data/samusik/`:

```bash
python3 run_samusik_benchmark.py
python3 run_samusik_full_python_fast_cached.py
python3 run_samusik_full_python_tessellation.py
```

Generated data and benchmark outputs are ignored by git.
