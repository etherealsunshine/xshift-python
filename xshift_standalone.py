"""Standalone Python wrapper for the VorteX ``standalone/Xshift.java`` flow.

This mirrors the Java command-line wrapper:

* read ``fcsFileList.txt``
* read ``importConfig.txt``
* import and transform events through the same rules as ``DatasetImporter``
* run X-shift with angular distance
* export per-input-file cluster assignments and an MST graph

FCS support is optional because Python FCS packages are not part of the standard
library. Reading uses ``fcsparser`` first, then ``flowio``. Writing FCS requires
``fcswrite``; without it, the wrapper writes CSV files with the same events plus
``cluster_id``.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from xshift import XShift


@dataclass
class ImportConfig:
    dataset_name: str = "default_ds"
    clustering_columns: tuple[str, ...] = ("",)
    side_columns: tuple[str, ...] = ("",)
    rescale: str = "NONE"
    transformation: str = "ASINH"
    scaling_factor: float = 5.0
    quantile: float = 0.95
    rescale_separately: bool = False
    limit_events_per_file: int = 0
    euclidean_len_ths: float = 0.0
    noise_threshold: float = 0.0


@dataclass
class EventTable:
    path: Path
    data: pd.DataFrame
    short_names: list[str]
    long_names: list[str]

    @property
    def row_count(self) -> int:
        return int(self.data.shape[0])

    @property
    def file_name_no_ext(self) -> str:
        return self.path.stem


@dataclass
class ImportedDataset:
    X: np.ndarray
    file_ids: np.ndarray
    row_indices: np.ndarray
    row_names: list[str]
    feature_names: list[str]
    skipped_rows: dict[str, list[int]]


def read_java_properties(path: Path) -> dict[str, str]:
    props: dict[str, str] = {}
    pending = ""
    for raw_line in path.read_text().splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith(("#", "!")):
            continue
        if line.endswith("\\"):
            pending += line[:-1]
            continue
        line = pending + line
        pending = ""
        sep_positions = [pos for pos in (line.find("="), line.find(":")) if pos >= 0]
        if sep_positions:
            pos = min(sep_positions)
            key = line[:pos].strip()
            value = line[pos + 1 :].strip()
        else:
            parts = line.split(None, 1)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ""
        props[key] = value
    return props


def read_import_config(path: Path) -> ImportConfig:
    p = read_java_properties(path)
    return ImportConfig(
        clustering_columns=tuple(p.get("clustering_columns", "").split(";")),
        side_columns=tuple(p.get("side_columns", "").split(";")),
        rescale=p.get("rescale", "NONE").upper(),
        transformation=p.get("transformation", "ASINH").upper(),
        scaling_factor=float(p.get("scaling_factor", "5")),
        quantile=float(p.get("quantile", "0.95")),
        rescale_separately=p.get("rescale_separately", "false").lower() == "true",
        limit_events_per_file=int(p.get("limit_events_per_file", "0")),
        euclidean_len_ths=float(p.get("euclidian_length_threshold", "0.0")),
        noise_threshold=float(p.get("noise_threshold", "0")),
    )


def read_file_list(path: Path) -> list[Path]:
    base = path.parent
    files = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            p = Path(line)
            files.append(p if p.is_absolute() else base / p)
    return files


def load_event_table(path: Path) -> EventTable:
    suffix = path.suffix.lower()
    if suffix == ".fcs":
        return load_fcs(path)
    data = pd.read_csv(path)
    names = list(data.columns)
    return EventTable(path=path, data=data, short_names=names, long_names=names)


def load_fcs(path: Path) -> EventTable:
    try:
        from fcsparser import parse

        meta, data = parse(str(path), reformat_meta=True)
        short_names = list(data.columns)
        long_names = [str(meta.get(f"$P{i + 1}S", short_names[i]) or short_names[i]) for i in range(len(short_names))]
        return EventTable(path=path, data=data.reset_index(drop=True), short_names=short_names, long_names=long_names)
    except ImportError:
        pass

    try:
        import flowio
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot read FCS file {path}. Install fcsparser or flowio, or use CSV input."
        ) from exc

    fd = flowio.FlowData(str(path))
    n_channels = int(fd.channel_count)
    arr = np.asarray(fd.events, dtype=float).reshape((-1, n_channels))
    short_names = [fd.channels[str(i + 1)]["PnN"] for i in range(n_channels)]
    long_names = [fd.channels[str(i + 1)].get("PnS") or short_names[i] for i in range(n_channels)]
    return EventTable(path=path, data=pd.DataFrame(arr, columns=short_names), short_names=short_names, long_names=long_names)


def resolve_feature_columns(config: ImportConfig, first: EventTable) -> list[str]:
    raw = config.clustering_columns[0].strip() if config.clustering_columns else ""
    if not raw:
        raise ValueError("importConfig.txt must define clustering_columns")
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    resolved = []
    for token in tokens:
        if token.isdigit():
            idx = int(token) - 1
            if idx < 0 or idx >= len(first.short_names):
                raise IndexError(f"clustering column index {token} is outside available channels")
            resolved.append(first.short_names[idx])
        else:
            matches = [name for name in first.short_names if name.lower() == token.lower()]
            if not matches:
                raise ValueError(f"Invalid clustering channel {token}; available: {first.short_names}")
            resolved.append(matches[0])
    return resolved


def import_dataset(
    tables: Sequence[EventTable],
    config: ImportConfig,
    feature_columns: Sequence[str],
    *,
    rng: np.random.Generator,
) -> ImportedDataset:
    sigma = math.sqrt(config.euclidean_len_ths / (len(feature_columns) * 0.33)) / 100 if config.euclidean_len_ths > 0 else 0.001
    rows: list[np.ndarray] = []
    file_ids: list[int] = []
    row_indices: list[int] = []
    row_names: list[str] = []
    skipped_rows: dict[str, list[int]] = {}

    for file_id, table in enumerate(tables):
        short_lower = {name.lower(): name for name in table.short_names}
        selected = []
        for col in feature_columns:
            match = short_lower.get(col.lower())
            if match is None:
                raise ValueError(f"{table.path} is missing clustering channel {col}")
            selected.append(match)

        imported_for_file = 0
        skipped_for_file: list[int] = []
        values = table.data[selected].to_numpy(dtype=float)
        for row_idx, vec in enumerate(values):
            if not np.isfinite(vec).all():
                skipped_for_file.append(row_idx)
                continue
            data_vec = vec.astype(float, copy=True)
            data_vec = transform_vector(data_vec, config)
            if config.noise_threshold > 0:
                zero_mask = data_vec == 0
                data_vec[zero_mask] = rng.normal(0.0, sigma, size=int(zero_mask.sum()))
            if np.linalg.norm(data_vec) > config.euclidean_len_ths:
                rows.append(data_vec)
                file_ids.append(file_id)
                row_indices.append(row_idx)
                row_names.append(f"{table.file_name_no_ext}_{row_idx}")
                imported_for_file += 1
                if config.limit_events_per_file > 0 and imported_for_file >= config.limit_events_per_file:
                    break
        skipped_rows[str(table.path)] = skipped_for_file

    if not rows:
        raise ValueError("No datapoints were imported after transformation/filtering")

    X = np.vstack(rows)
    X = rescale_matrix(X, config)
    return ImportedDataset(
        X=X,
        file_ids=np.asarray(file_ids, dtype=int),
        row_indices=np.asarray(row_indices, dtype=int),
        row_names=row_names,
        feature_names=list(feature_columns),
        skipped_rows=skipped_rows,
    )


def transform_vector(vec: np.ndarray, config: ImportConfig) -> np.ndarray:
    if config.noise_threshold > 0:
        signs = np.sign(vec)
        vec = signs * np.maximum(vec - config.noise_threshold, 0.0)
    if config.transformation == "ASINH":
        vec = np.arcsinh(vec / config.scaling_factor)
    elif config.transformation == "DOUBLE_ASINH":
        vec = np.arcsinh(np.arcsinh(vec / config.scaling_factor))
    elif config.transformation != "NONE":
        raise ValueError(f"Unsupported transformation {config.transformation}")
    return vec


def rescale_matrix(X: np.ndarray, config: ImportConfig) -> np.ndarray:
    if config.rescale == "NONE":
        return X
    if config.rescale == "QUANTILE":
        idx = int(math.floor(config.quantile * X.shape[0]))
        idx = int(np.clip(idx, 0, X.shape[0] - 1))
        scale = np.sort(X, axis=0)[idx]
    elif config.rescale == "SD":
        scale = X.std(axis=0, ddof=1)
    else:
        raise ValueError(f"Unsupported rescale mode {config.rescale}")
    scale = np.where(scale == 0, 1.0, scale)
    return X / scale


def assign_full_file_labels(
    table: EventTable,
    file_id: int,
    dataset: ImportedDataset,
    labels: np.ndarray,
    X_full: np.ndarray,
    X_clustered: np.ndarray,
    *,
    sample_size: int = 1000,
    rng: np.random.Generator,
) -> np.ndarray:
    cluster_id_map = np.full(table.row_count, -1, dtype=int)
    in_file = dataset.file_ids == file_id
    cluster_id_map[dataset.row_indices[in_file]] = labels[in_file]

    missing = np.flatnonzero(cluster_id_map < 0)
    if missing.size == 0:
        return cluster_id_map

    reps: dict[int, np.ndarray] = {}
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if idx.size > sample_size:
            idx = rng.choice(idx, size=sample_size, replace=False)
        reps[int(label)] = X_clustered[idx]

    for row_idx in missing:
        vec = X_full[row_idx]
        best_label = -1
        best_cos = -np.inf
        for label, rep in reps.items():
            denom = np.linalg.norm(rep, axis=1) * max(float(np.linalg.norm(vec)), 1e-12)
            cos = (rep @ vec) / np.where(denom == 0, 1.0, denom)
            max_cos = float(np.max(cos))
            if max_cos > best_cos:
                best_cos = max_cos
                best_label = label
        cluster_id_map[row_idx] = best_label
    return cluster_id_map


def transformed_full_matrix(table: EventTable, feature_columns: Sequence[str], config: ImportConfig) -> np.ndarray:
    short_lower = {name.lower(): name for name in table.short_names}
    selected = [short_lower[col.lower()] for col in feature_columns]
    X = table.data[selected].to_numpy(dtype=float)
    X = np.vstack([transform_vector(row.astype(float, copy=True), config) for row in X])
    return rescale_matrix(X, config)


def export_clustered_file(
    table: EventTable,
    labels: np.ndarray,
    out_dir: Path,
    cluster_name: str,
    *,
    write_fcs: bool,
) -> Path:
    out = table.data.copy()
    out["cluster_id"] = labels.astype(int)

    if write_fcs and table.path.suffix.lower() == ".fcs":
        try:
            from fcswrite import write_fcs

            out_path = out_dir / table.path.name
            channel_names = table.short_names + [cluster_name]
            write_fcs(str(out_path), channel_names, out.to_numpy(dtype=np.float32))
            return out_path
        except ImportError:
            pass

    out_path = out_dir / f"{table.path.stem}_clustered.csv"
    out.to_csv(out_path, index=False)
    return out_path


def write_mst_gml(path: Path, centers: np.ndarray, sizes: np.ndarray) -> None:
    if centers.shape[0] == 0:
        path.write_text("graph [\n  directed 0\n]\n")
        return
    edges = angular_maximum_spanning_tree(centers)

    lines = ["graph [", "  directed 0"]
    for idx, (center, size) in enumerate(zip(centers, sizes)):
        lines.extend(
            [
                "  node [",
                f"    id {idx}",
                f'    label "cluster_{idx}"',
                f"    size {int(size)}",
                f'    center "{",".join(f"{v:.8g}" for v in center)}"',
                "  ]",
            ]
        )
    for source, target, weight in edges:
        lines.extend(
            [
                "  edge [",
                f"    source {int(source)}",
                f"    target {int(target)}",
                f"    weight {float(weight):.8g}",
                "  ]",
            ]
        )
    lines.append("]")
    path.write_text("\n".join(lines) + "\n")


def angular_maximum_spanning_tree(centers: np.ndarray) -> list[tuple[int, int, float]]:
    n = centers.shape[0]
    if n <= 1:
        return []
    norms = np.linalg.norm(centers, axis=1)
    denom = np.outer(np.where(norms == 0, 1.0, norms), np.where(norms == 0, 1.0, norms))
    sim = (centers @ centers.T) / denom
    max_sim = float(np.nanmax(sim[np.triu_indices(n, k=1)]))
    if not np.isfinite(max_sim) or max_sim == 0:
        max_sim = 1.0

    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            candidates.append((float(sim[i, j]), i, j))
    candidates.sort(reverse=True)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    kept: list[tuple[int, int, float]] = []
    for similarity, i, j in candidates:
        ri = find(i)
        rj = find(j)
        if ri == rj:
            continue
        parent[ri] = rj
        # MSTBuilder scales by 5, restores kept edges with another factor of 5,
        # then buildGraph multiplies final edge weights by 3.
        kept.append((i, j, 75.0 * similarity / max_sim))
        if len(kept) == n - 1:
            break
    return kept


def run_xshift_wrapper(
    k_arg: str = "20",
    config_path: Path = Path("importConfig.txt"),
    fcs_list_path: Path = Path("fcsFileList.txt"),
    output_path: Path = Path("out"),
    *,
    seed: int = 0,
    write_fcs: bool = True,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    config = read_import_config(config_path)
    files = read_file_list(fcs_list_path)
    if not files:
        raise ValueError(f"No files listed in {fcs_list_path}")

    tables = [load_event_table(path) for path in files]
    feature_columns = resolve_feature_columns(config, tables[0])
    print("Clustering columns:")
    for col in feature_columns:
        idx = tables[0].short_names.index(col)
        print(f"  {idx + 1}={tables[0].long_names[idx]}|{tables[0].short_names[idx]}")

    dataset = import_dataset(tables, config, feature_columns, rng=rng)
    print(f"Imported {dataset.X.shape[0]} events across {len(tables)} file(s)")

    model = XShift(k=20 if k_arg == "auto" else int(k_arg), metric="angular")
    if k_arg == "auto" or int(k_arg) < 3:
        dim = dataset.X.shape[1]
        max_k = int(2890 * (dim**-0.8))
        min_k = int(1942 * (dim**-1.61))
        result = model.fit_auto(dataset.X, max_k, min_k, 30)
    else:
        result = model.fit(dataset.X).result_
    print(f"X-shift produced {result.n_clusters} clusters with K={result.k}")

    cluster_name = f"X-shift K={result.k}"
    for file_id, table in enumerate(tables):
        full_X = transformed_full_matrix(table, feature_columns, config)
        file_labels = assign_full_file_labels(
            table,
            file_id,
            dataset,
            result.labels,
            full_X,
            dataset.X,
            rng=rng,
        )
        out_file = export_clustered_file(table, file_labels, output_path, cluster_name, write_fcs=write_fcs)
        print(f"Wrote {out_file}")

    write_mst_gml(output_path / "mst.gml", result.centers, result.cluster_sizes)
    print(f"Wrote {output_path / 'mst.gml'}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Python port of standalone/Xshift.java")
    parser.add_argument("k", nargs="?", default="20", help="NUM_NEAREST_NEIGHBORS or auto")
    parser.add_argument("--config", type=Path, default=Path("importConfig.txt"))
    parser.add_argument("--file-list", type=Path, default=Path("fcsFileList.txt"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv-only", action="store_true", help="Always write CSV outputs, even for FCS inputs.")
    args = parser.parse_args(argv)

    run_xshift_wrapper(
        args.k,
        args.config,
        args.file_list,
        args.out,
        seed=args.seed,
        write_fcs=not args.csv_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
