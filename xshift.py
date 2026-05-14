"""Python port of the VorteX/X-shift cytometry clustering core.

The original Java entry point at ``standalone/Xshift.java`` is mostly file I/O
and delegates the clustering work to ``vortex.clustering.XShiftClustering``.
This module ports that clustering path into a small Python API that accepts a
numeric events x markers matrix.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors


Metric = Literal["angular", "euclidean"]
ValleyKnnBackend = Literal["sklearn", "matrix", "hnsw"]
MainKnnBackend = Literal["sklearn", "tessellation"]


@dataclass
class XShiftResult:
    labels: np.ndarray
    densities: np.ndarray
    parents: np.ndarray
    modes: np.ndarray
    centers: np.ndarray
    cluster_sizes: np.ndarray
    k: int
    metric: Metric

    @property
    def n_clusters(self) -> int:
        return int(self.cluster_sizes.size)


class XShift:
    """Density-gradient clustering inspired by Nolan Lab VorteX X-shift.

    Parameters
    ----------
    k:
        Number of nearest neighbors used for the density estimate. This is the
        command-line ``K`` in the Java wrapper.
    n_size:
        Number of nearest neighbors searched for a higher-density parent. The
        Java UI supplies this separately; using ``k`` is a good default.
    metric:
        ``"angular"`` mirrors the standalone Java wrapper's default
        ``AngularDistance`` intent for cytometry profiles. ``"euclidean"`` is
        also available.
    merge_mahalanobis:
        Final greedy merge threshold. The Java implementation uses ``2.0``.
        Set to ``None`` or ``0`` to skip this final merge.
    root_merge:
        Whether to merge local roots if the density valley between roots does
        not drop below the lower-density root. This follows the Java code's
        Gabriel-neighbor/root filtering stage.
    random_state:
        Present for API symmetry; exact-neighbor clustering is deterministic.
    """

    def __init__(
        self,
        k: int = 20,
        n_size: int | None = None,
        metric: Metric = "angular",
        merge_mahalanobis: float | None = 2.0,
        root_merge: bool = True,
        root_merge_step: float = 0.05,
        random_state: int | None = None,
        verbose: bool = False,
        query_chunk_size: int = 512,
        valley_knn_backend: ValleyKnnBackend = "sklearn",
        main_knn_backend: MainKnnBackend = "sklearn",
        hnsw_ef: int = 80,
        hnsw_m: int = 16,
    ) -> None:
        self.k = int(k)
        self.n_size = None if n_size is None else int(n_size)
        self.metric = metric
        self.merge_mahalanobis = merge_mahalanobis
        self.root_merge = bool(root_merge)
        self.root_merge_step = float(root_merge_step)
        self.random_state = random_state
        self.verbose = verbose
        self.query_chunk_size = int(query_chunk_size)
        self.valley_knn_backend = valley_knn_backend
        self.main_knn_backend = main_knn_backend
        self.hnsw_ef = int(hnsw_ef)
        self.hnsw_m = int(hnsw_m)

    def fit(self, X: np.ndarray) -> "XShift":
        X = self._validate_X(X)
        self._log(f"fit start: n={X.shape[0]}, dim={X.shape[1]}, k={self.k}, metric={self.metric}")
        self._neighbor_X = X
        self._neighbor_model = self._fit_neighbor_model(X)
        self._hnsw_index = self._fit_hnsw_index(X) if self.valley_knn_backend == "hnsw" else None
        self._log("nearest-neighbor index ready")
        n = X.shape[0]
        if not 1 <= self.k < n:
            raise ValueError(f"k must be between 1 and n_samples - 1; got {self.k} for {n} samples")

        n_size = self._java_auto_n_size(X.shape[0], X.shape[1]) if self.n_size is None else self.n_size
        n_size = int(np.clip(n_size, 2, n))
        max_neighbors = max(self.k, n_size)

        if self.main_knn_backend == "tessellation":
            neighbor_idx, neighbor_dist = self._tessellated_angular_knn_graph(X, max_neighbors)
        else:
            neighbor_idx, neighbor_dist = self._nearest_neighbors(X, X, max_neighbors)
        self._log(f"computed {max_neighbors}-NN graph")
        densities = -neighbor_dist[:, : self.k].sum(axis=1)
        parents = self._higher_density_parents(neighbor_idx, neighbor_dist, densities, n_size)
        self._log(f"initial roots: {int(np.sum(parents == -1))}")

        if self.root_merge:
            parents = self._merge_roots(X, densities, parents)
            self._log(f"roots after root merge: {int(np.sum(parents == -1))}")

        labels, modes = self._assign_by_gradient(parents)
        self._log(f"clusters after gradient assignment: {int(np.unique(labels).size)}")
        if self.merge_mahalanobis and self.merge_mahalanobis > 0:
            labels = self._merge_by_diagonal_mahalanobis(X, labels, float(self.merge_mahalanobis))
            labels, modes = self._relabel(labels, parents)
            self._log(f"clusters after Mahalanobis merge: {int(np.unique(labels).size)}")

        centers, sizes = self._cluster_centers(X, labels)
        self.result_ = XShiftResult(
            labels=labels,
            densities=densities,
            parents=parents,
            modes=modes,
            centers=centers,
            cluster_sizes=sizes,
            k=self.k,
            metric=self.metric,
        )
        self.labels_ = labels
        self.densities_ = densities
        self.parents_ = parents
        self.cluster_centers_ = centers
        self._log("fit done")
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    def fit_auto(
        self,
        X: np.ndarray,
        from_k: int,
        to_k: int,
        steps: int = 30,
    ) -> XShiftResult:
        """Try a Java-style square-spaced K range and keep an elbow solution."""
        if steps < 1:
            raise ValueError("steps must be at least 1")
        from_root = np.sqrt(from_k)
        to_root = np.sqrt(to_k)
        grid = np.linspace(from_root, to_root, steps)
        k_values = np.unique(np.maximum(1, grid.astype(float) ** 2).astype(int))
        k_values = k_values[(k_values >= 1) & (k_values < np.asarray(X).shape[0])]
        if k_values.size == 0:
            raise ValueError("automatic K range did not contain any valid K values")

        results: list[XShiftResult] = []
        counts: list[int] = []
        original_k = self.k
        for k in k_values:
            self.k = int(k)
            result = self.fit(X).result_
            results.append(result)
            counts.append(result.n_clusters)
        self.k = original_k

        best = _elbow_index(np.asarray(k_values, dtype=float), np.asarray(counts, dtype=float))
        self.result_ = results[best]
        self.labels_ = self.result_.labels
        self.densities_ = self.result_.densities
        self.parents_ = self.result_.parents
        self.cluster_centers_ = self.result_.centers
        return self.result_

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array shaped (events, markers)")
        if X.shape[0] < 2:
            raise ValueError("X must contain at least two events")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values")
        if self.metric not in ("angular", "euclidean"):
            raise ValueError("metric must be 'angular' or 'euclidean'")
        return X

    def _nearest_neighbors(
        self,
        X: np.ndarray,
        query: np.ndarray,
        n_neighbors: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_neighbors = min(n_neighbors, X.shape[0])
        if self.metric == "angular" and getattr(self, "_neighbor_X", None) is X and query is not X and self.valley_knn_backend == "hnsw":
            return self._angular_knn_hnsw(X, query, n_neighbors)
        if self.metric == "angular" and getattr(self, "_neighbor_X", None) is X and query is not X and self.valley_knn_backend == "matrix":
            return self._angular_knn_bruteforce(X, query, n_neighbors)
        nn = self._neighbor_model if getattr(self, "_neighbor_X", None) is X else self._fit_neighbor_model(X)
        if self.metric == "angular":
            Xn = _unit(X)
            Qn = _unit(query)
            _, idx = nn.kneighbors(Qn, n_neighbors=n_neighbors, return_distance=True)
            dist = self._pairwise_dist_rows(query, X[idx])
            order = np.argsort(dist, axis=1, kind="mergesort")
            idx = np.take_along_axis(idx, order, axis=1)
            dist = np.take_along_axis(dist, order, axis=1)
            return idx, dist

        dist, idx = nn.kneighbors(query, n_neighbors=n_neighbors, return_distance=True)
        return idx, dist

    def _angular_knn_bruteforce(
        self,
        X: np.ndarray,
        query: np.ndarray,
        n_neighbors: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        Xn = _unit(X)
        Qn = _unit(query)
        all_idx = np.empty((query.shape[0], n_neighbors), dtype=int)
        all_dist = np.empty((query.shape[0], n_neighbors), dtype=float)
        for start in range(0, query.shape[0], self.query_chunk_size):
            stop = min(start + self.query_chunk_size, query.shape[0])
            sim = Qn[start:stop] @ Xn.T
            part = np.argpartition(-sim, kth=n_neighbors - 1, axis=1)[:, :n_neighbors]
            part_sim = np.take_along_axis(sim, part, axis=1)
            dist = np.arccos(np.clip(part_sim, -1.0, 1.0)) / np.pi
            q_norm = np.linalg.norm(query[start:stop], axis=1)
            x_norm = np.linalg.norm(X[part], axis=2)
            dist[(q_norm[:, None] == 0) | (x_norm == 0)] = 0.0
            order = np.argsort(dist, axis=1, kind="mergesort")
            all_idx[start:stop] = np.take_along_axis(part, order, axis=1)
            all_dist[start:stop] = np.take_along_axis(dist, order, axis=1)
        return all_idx, all_dist

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] XShift: {message}", flush=True)

    def _fit_neighbor_model(self, X: np.ndarray) -> NearestNeighbors:
        if self.metric == "angular":
            nn = NearestNeighbors(metric="cosine", algorithm="brute")
            nn.fit(_unit(X))
            return nn
        nn = NearestNeighbors(metric="euclidean", algorithm="auto")
        nn.fit(X)
        return nn

    def _fit_hnsw_index(self, X: np.ndarray):
        if self.metric != "angular":
            raise ValueError("hnsw valley backend currently supports only angular metric")
        try:
            import hnswlib
        except ImportError as exc:
            raise ImportError("Install hnswlib or use valley_knn_backend='sklearn'") from exc
        Xn = _unit(X).astype(np.float32, copy=False)
        index = hnswlib.Index(space="cosine", dim=X.shape[1])
        index.init_index(max_elements=X.shape[0], ef_construction=max(100, self.hnsw_ef * 2), M=self.hnsw_m)
        index.add_items(Xn, np.arange(X.shape[0]))
        index.set_ef(max(self.hnsw_ef, self.k))
        return index

    def _angular_knn_hnsw(
        self,
        X: np.ndarray,
        query: np.ndarray,
        n_neighbors: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._hnsw_index is None:
            return self._angular_knn_bruteforce(X, query, n_neighbors)
        idx, _ = self._hnsw_index.knn_query(_unit(query).astype(np.float32, copy=False), k=n_neighbors)
        dist = self._pairwise_dist_rows(query, X[idx])
        order = np.argsort(dist, axis=1, kind="mergesort")
        idx = np.take_along_axis(idx, order, axis=1)
        dist = np.take_along_axis(dist, order, axis=1)
        return idx, dist

    def _tessellated_angular_knn_graph(self, X: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
        if self.metric != "angular":
            raise ValueError("tessellation main KNN backend currently supports only angular metric")
        n = X.shape[0]
        n_neighbors = min(n_neighbors, n)
        num_cells = max(1, int(np.sqrt(n)))
        cells, centers = self._split_dataset_into_cells(X, num_cells)
        Xn = _unit(X)
        idx = np.empty((n, n_neighbors), dtype=int)
        dist = np.empty((n, n_neighbors), dtype=float)
        self._log(f"tessellation: {len(cells)} non-empty cells")
        last_log = time.time()
        done = 0
        for cell_id, point_idx in cells:
            center = centers[cell_id]
            center_dist = self._angular_distance_one_to_many(center, X)
            central_order = np.argsort(center_dist, kind="mergesort")
            central_dist = center_dist[central_order]
            cell_idx, cell_dist = self._tessellated_knn_for_cell(
                X[point_idx],
                X,
                Xn,
                central_order,
                central_dist,
                center,
                n_neighbors,
            )
            idx[point_idx] = cell_idx
            dist[point_idx] = cell_dist
            done += len(point_idx)
            if self.verbose and time.time() - last_log > 10:
                self._log(f"tessellation KNN progress: {done}/{n}")
                last_log = time.time()
        return idx, dist

    def _split_dataset_into_cells(self, X: np.ndarray, num_cells: int) -> tuple[list[tuple[int, np.ndarray]], np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        num_cells = min(num_cells, n)
        seed_idx = rng.permutation(n)[:num_cells]
        centers = X[seed_idx].astype(float, copy=True)
        Xn = _unit(X)
        centers_n = _unit(centers)
        assignments = np.argmax(Xn @ centers_n.T, axis=1)

        new_centers = np.zeros_like(centers)
        sizes = np.bincount(assignments, minlength=num_cells)
        np.add.at(new_centers, assignments, Xn)
        non_empty = sizes > 0
        centers[non_empty] = _unit(new_centers[non_empty])

        cells: list[tuple[int, np.ndarray]] = []
        for cell_id in range(num_cells):
            point_idx = np.flatnonzero(assignments == cell_id)
            if point_idx.size:
                cells.append((cell_id, point_idx))
        return cells, centers

    def _tessellated_knn_for_cell(
        self,
        points: np.ndarray,
        X: np.ndarray,
        Xn: np.ndarray,
        central_order: np.ndarray,
        central_dist: np.ndarray,
        center: np.ndarray,
        n_neighbors: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        first = central_order[:n_neighbors]
        points_n = _unit(points)
        initial_dist = np.arccos(np.clip(points_n @ Xn[first].T, -1.0, 1.0)) / np.pi
        zero_points = np.linalg.norm(points, axis=1) == 0
        if np.any(zero_points):
            initial_dist[zero_points] = 0.0
        initial_max = np.partition(initial_dist, n_neighbors - 1, axis=1)[:, n_neighbors - 1]
        dist_to_center = self._angular_distance_many_to_one(points, center)
        end_pos = np.searchsorted(central_dist, dist_to_center + initial_max, side="right")
        end_pos = np.maximum(end_pos, n_neighbors)
        max_end = int(end_pos.max())

        candidates = central_order[:max_end]
        cand_dist = np.arccos(np.clip(points_n @ Xn[candidates].T, -1.0, 1.0)) / np.pi
        if np.any(zero_points):
            cand_dist[zero_points] = 0.0
        mask = np.arange(max_end)[None, :] >= end_pos[:, None]
        cand_dist[mask] = np.inf

        part = np.argpartition(cand_dist, kth=n_neighbors - 1, axis=1)[:, :n_neighbors]
        part_dist = np.take_along_axis(cand_dist, part, axis=1)
        order = np.argsort(part_dist, axis=1, kind="mergesort")
        part = np.take_along_axis(part, order, axis=1)
        part_dist = np.take_along_axis(part_dist, order, axis=1)
        return candidates[part], part_dist

    def _angular_distance_pair(self, a: np.ndarray, b: np.ndarray) -> float:
        an = float(np.linalg.norm(a))
        bn = float(np.linalg.norm(b))
        if an == 0 or bn == 0:
            return 0.0
        return float(np.arccos(np.clip(np.dot(a, b) / (an * bn), -1.0, 1.0)) / np.pi)

    def _angular_distance_one_to_many(self, a: np.ndarray, B: np.ndarray) -> np.ndarray:
        an = float(np.linalg.norm(a))
        bn = np.linalg.norm(B, axis=1)
        if an == 0:
            return np.zeros(B.shape[0], dtype=float)
        dots = (B @ a) / np.where(bn == 0, 1.0, bn) / an
        dist = np.arccos(np.clip(dots, -1.0, 1.0)) / np.pi
        dist[bn == 0] = 0.0
        return dist

    def _angular_distance_many_to_one(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        bn = float(np.linalg.norm(b))
        an = np.linalg.norm(A, axis=1)
        if bn == 0:
            return np.zeros(A.shape[0], dtype=float)
        dots = (A @ b) / np.where(an == 0, 1.0, an) / bn
        dist = np.arccos(np.clip(dots, -1.0, 1.0)) / np.pi
        dist[an == 0] = 0.0
        return dist

    def _pairwise_dist_rows(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if B.ndim == 3:
            if self.metric == "euclidean":
                return np.linalg.norm(B - A[:, None, :], axis=2)
            a_norm = np.linalg.norm(A, axis=1)
            b_norm = np.linalg.norm(B, axis=2)
            A_unit = A / np.where(a_norm[:, None] == 0, 1.0, a_norm[:, None])
            B_unit = B / np.where(b_norm[:, :, None] == 0, 1.0, b_norm[:, :, None])
            dots = np.einsum("ij,ikj->ik", A_unit, B_unit)
            dist = np.arccos(np.clip(dots, -1.0, 1.0)) / np.pi
            dist[(a_norm[:, None] == 0) | (b_norm == 0)] = 0.0
            return dist

        if self.metric == "euclidean":
            return np.linalg.norm(A - B, axis=1)
        a_norm = np.linalg.norm(A, axis=1)
        b_norm = np.linalg.norm(B, axis=1)
        dots = np.sum(_unit(A) * _unit(B), axis=1)
        dist = np.arccos(np.clip(dots, -1.0, 1.0)) / np.pi
        dist[(a_norm == 0) | (b_norm == 0)] = 0.0
        return dist

    def _dist_one_to_many(self, x: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x2 = np.repeat(x[None, :], Y.shape[0], axis=0)
        return self._pairwise_dist_rows(x2, Y)

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric == "angular":
            return float(np.dot(_unit(a[None, :])[0], _unit(b[None, :])[0]))
        return float(1.0 / (self._dist_one_to_many(a, b[None, :])[0] + 1.0))

    def _similarities(self, a: np.ndarray, B: np.ndarray) -> np.ndarray:
        if self.metric == "angular":
            return _unit(B) @ _unit(a[None, :])[0]
        return 1.0 / (self._dist_one_to_many(a, B) + 1.0)

    def _distance_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if self.metric == "euclidean":
            return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        a_norm = np.linalg.norm(A, axis=1)
        b_norm = np.linalg.norm(B, axis=1)
        Au = A / np.where(a_norm[:, None] == 0, 1.0, a_norm[:, None])
        Bu = B / np.where(b_norm[:, None] == 0, 1.0, b_norm[:, None])
        dist = np.arccos(np.clip(Au @ Bu.T, -1.0, 1.0)) / np.pi
        dist[(a_norm[:, None] == 0) | (b_norm[None, :] == 0)] = 0.0
        return dist

    @staticmethod
    def _java_auto_n_size(n_samples: int, n_dimensions: int, p_value: float = 0.01) -> int:
        return int(max(0.5 * (n_dimensions + 1), -int(np.ceil(np.log(p_value / n_samples) / np.log(2)))))

    def _higher_density_parents(
        self,
        neighbor_idx: np.ndarray,
        neighbor_dist: np.ndarray,
        densities: np.ndarray,
        n_size: int,
    ) -> np.ndarray:
        parents = np.full(neighbor_idx.shape[0], -1, dtype=int)
        for i in range(neighbor_idx.shape[0]):
            candidates = neighbor_idx[i, 1:n_size]
            distances = neighbor_dist[i, 1:n_size]
            higher = densities[candidates] > densities[i]
            if np.any(higher):
                best = np.argmin(np.where(higher, distances, np.inf))
                parents[i] = int(candidates[best])
        return parents

    def _merge_roots(self, X: np.ndarray, densities: np.ndarray, parents: np.ndarray) -> np.ndarray:
        roots = np.flatnonzero(parents == -1)
        if roots.size <= 1:
            return parents

        root_X = X[roots]
        updated = parents.copy()
        steps = np.arange(self.root_merge_step, 1.0, self.root_merge_step)
        last_log = time.time()

        for root_i, root in enumerate(roots):
            if self.verbose and (root_i == 0 or root_i == roots.size - 1 or time.time() - last_log > 10):
                self._log(f"root merge progress: {root_i + 1}/{roots.size}")
                last_log = time.time()
            eligible = roots[(roots != root) & (densities[root] <= densities[roots])]
            if eligible.size == 0:
                continue

            mids = self._midpoints(X[root], X[eligible])
            nearest_roots = roots[np.argsort(self._distance_matrix(mids, root_X), axis=1, kind="mergesort")[:, :2]]
            gabriel = np.array(
                [set(pair.tolist()) == {int(root), int(other)} for pair, other in zip(nearest_roots, eligible)],
                dtype=bool,
            )
            candidates = eligible[gabriel]
            if candidates.size == 0:
                continue

            if steps.size:
                path_points = self._weighted_path_points(X[root], X[candidates], steps)
                _, path_dist = self._nearest_neighbors(X, path_points, self.k)
                path_density = -path_dist[:, : self.k].sum(axis=1).reshape(candidates.size, steps.size)
                candidates = candidates[np.all(path_density >= densities[root], axis=1)]
                if candidates.size == 0:
                    continue

            best_root = int(candidates[int(np.argmax(self._similarities(X[root], X[candidates])))])
            if updated[best_root] != root:
                updated[root] = best_root
        return updated

    def _midpoint(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.metric == "angular":
            return _unit(((_unit(a[None, :])[0] + _unit(b[None, :])[0])[None, :]))[0]
        return (a + b) * 0.5

    def _midpoints(self, a: np.ndarray, B: np.ndarray) -> np.ndarray:
        if self.metric == "angular":
            return _unit(_unit(a[None, :])[0][None, :] + _unit(B))
        return (a[None, :] + B) * 0.5

    def _weighted_average(self, a: np.ndarray, b: np.ndarray, weight: float) -> np.ndarray:
        if self.metric == "angular":
            return weight * _unit(a[None, :])[0] + (1.0 - weight) * _unit(b[None, :])[0]
        return weight * a + (1.0 - weight) * b

    def _weighted_path_points(self, a: np.ndarray, B: np.ndarray, steps: np.ndarray) -> np.ndarray:
        if self.metric == "angular":
            au = _unit(a[None, :])[0]
            bu = _unit(B)
            points = steps[:, None, None] * au[None, None, :] + (1.0 - steps[:, None, None]) * bu[None, :, :]
        else:
            points = steps[:, None, None] * a[None, None, :] + (1.0 - steps[:, None, None]) * B[None, :, :]
        return np.swapaxes(points, 0, 1).reshape(-1, a.size)

    def _valley_is_dense_enough(
        self,
        X: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        min_density: float,
        steps: np.ndarray,
    ) -> bool:
        if steps.size == 0:
            return True
        mids = np.vstack([self._weighted_average(a, b, float(s)) for s in steps])
        _, dist = self._nearest_neighbors(X, mids, self.k)
        return bool(np.all(-dist[:, : self.k].sum(axis=1) >= min_density))

    def _assign_by_gradient(self, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        roots = np.flatnonzero(parents == -1)
        root_to_label = {int(root): label for label, root in enumerate(roots)}
        labels = np.empty(parents.size, dtype=int)
        modes = np.empty(parents.size, dtype=int)

        for i in range(parents.size):
            seen: set[int] = set()
            current = int(i)
            while parents[current] >= 0:
                if current in seen:
                    current = min(seen)
                    break
                seen.add(current)
                current = int(parents[current])
            modes[i] = current
            labels[i] = root_to_label.setdefault(current, len(root_to_label))
        labels = _compact_labels(labels)
        return labels, modes

    def _merge_by_diagonal_mahalanobis(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        clusters = {int(lab): np.flatnonzero(labels == lab) for lab in np.unique(labels)}
        stats = {lab: _ClusterStats.from_indices(X, idx) for lab, idx in clusters.items()}
        next_lab = max(clusters) + 1 if clusters else 0
        round_no = 0

        distances: dict[tuple[int, int], float] = {}
        active = sorted(clusters)
        for pos, a in enumerate(active):
            for b in active[pos + 1 :]:
                distances[(a, b)] = stats[a].diagonal_mahalanobis(stats[b])

        while len(active) > 1:
            under = [(d, pair) for pair, d in distances.items() if d < threshold]
            if not under:
                break
            _, (a, b) = min(under, key=lambda item: item[0])
            new_lab = next_lab
            next_lab += 1
            clusters[new_lab] = np.concatenate([clusters.pop(a), clusters.pop(b)])
            stats[new_lab] = stats.pop(a).merged(stats.pop(b))
            active = [lab for lab in active if lab not in (a, b)]
            keys_to_drop = [pair for pair in distances if a in pair or b in pair]
            for pair in keys_to_drop:
                del distances[pair]
            for lab in active:
                pair = (lab, new_lab) if lab < new_lab else (new_lab, lab)
                distances[pair] = stats[lab].diagonal_mahalanobis(stats[new_lab])
            active.append(new_lab)
            round_no += 1
            if self.verbose and (round_no == 1 or round_no % 25 == 0):
                self._log(f"Mahalanobis merge round {round_no}, clusters left {len(active)}")

        merged = np.empty_like(labels)
        for label, lab in enumerate(active):
            idx = clusters[lab]
            merged[idx] = label
        return merged

    def _relabel(self, labels: np.ndarray, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        labels = _compact_labels(labels)
        modes = np.empty(labels.size, dtype=int)
        for label in np.unique(labels):
            idx = np.flatnonzero(labels == label)
            roots = idx[parents[idx] == -1]
            modes[idx] = int(roots[0] if roots.size else idx[0])
        return labels, modes

    @staticmethod
    def _cluster_centers(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        unique = np.unique(labels)
        centers = np.vstack([X[labels == lab].mean(axis=0) for lab in unique])
        sizes = np.asarray([np.sum(labels == lab) for lab in unique], dtype=int)
        return centers, sizes


def _unit(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norm == 0, 1.0, norm)


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    _, inv = np.unique(labels, return_inverse=True)
    return inv.astype(int)


@dataclass
class _ClusterStats:
    n: int
    sum_: np.ndarray
    sumsq: np.ndarray

    @classmethod
    def from_indices(cls, X: np.ndarray, indices: np.ndarray) -> "_ClusterStats":
        rows = X[indices]
        return cls(n=rows.shape[0], sum_=rows.sum(axis=0), sumsq=np.square(rows).sum(axis=0))

    def merged(self, other: "_ClusterStats") -> "_ClusterStats":
        return _ClusterStats(self.n + other.n, self.sum_ + other.sum_, self.sumsq + other.sumsq)

    @property
    def mean(self) -> np.ndarray:
        return self.sum_ / self.n

    @property
    def sd(self) -> np.ndarray:
        if self.n < 2:
            return np.full_like(self.sum_, np.inf, dtype=float)
        var = (self.sumsq - (self.sum_ * self.sum_) / self.n) / (self.n - 1)
        var = np.maximum(var, 0.0)
        return np.sqrt(var) / 1.03

    def diagonal_mahalanobis(self, other: "_ClusterStats") -> float:
        pooled = (self.sd + other.sd) * 0.5
        if np.any(pooled <= 0) or not np.isfinite(pooled).all():
            return np.inf
        return float(np.sqrt(np.sum(((self.mean - other.mean) / pooled) ** 2)))


def _diagonal_mahalanobis(A: np.ndarray, B: np.ndarray) -> float:
    if A.shape[0] < 2 or B.shape[0] < 2:
        return np.inf
    mean_a = A.mean(axis=0)
    mean_b = B.mean(axis=0)
    sd_a = A.std(axis=0, ddof=1) / 1.03
    sd_b = B.std(axis=0, ddof=1) / 1.03
    pooled = (sd_a + sd_b) * 0.5
    if np.any(pooled <= 0) or not np.isfinite(pooled).all():
        return np.inf
    return float(np.sqrt(np.sum(((mean_a - mean_b) / pooled) ** 2)))


def _elbow_index(x: np.ndarray, y: np.ndarray) -> int:
    if x.size <= 2:
        return 0
    pts = np.column_stack([x, y]).astype(float)
    mins = pts.min(axis=0)
    span = np.ptp(pts, axis=0)
    ranges = np.where(span == 0, 1.0, span)
    pts = (pts - mins) / ranges
    line = pts[-1] - pts[0]
    denom = np.linalg.norm(line)
    if denom == 0:
        return 0
    dist = np.abs(np.cross(line, pts - pts[0]) / denom)
    return int(np.argmax(dist))


def _parse_features(columns: Sequence[str], features: str | None) -> list[str]:
    if features is None:
        return list(columns)
    selected: list[str] = []
    for part in features.split(","):
        token = part.strip()
        if not token:
            continue
        if token.isdigit():
            selected.append(columns[int(token) - 1])
        else:
            selected.append(token)
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cluster cytometry-like tabular data with X-shift.")
    parser.add_argument("input", type=Path, help="CSV file with events as rows.")
    parser.add_argument("--features", help="Comma-separated column names or 1-based column indexes.")
    parser.add_argument("--k", default="20", help="Neighbor count or 'auto'. Default: 20.")
    parser.add_argument("--metric", choices=("angular", "euclidean"), default="angular")
    parser.add_argument("--n-size", type=int, default=None, help="Higher-density parent search neighborhood.")
    parser.add_argument("--output", type=Path, default=Path("xshift_clusters.csv"))
    parser.add_argument("--no-root-merge", action="store_true")
    parser.add_argument("--no-mahalanobis-merge", action="store_true")
    args = parser.parse_args(argv)

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("The CLI needs pandas. Install it or call XShift from Python with a NumPy array.") from exc

    df = pd.read_csv(args.input)
    feature_columns = _parse_features(list(df.columns), args.features)
    X = df[feature_columns].to_numpy(dtype=float)

    model = XShift(
        k=20 if args.k == "auto" else int(args.k),
        n_size=args.n_size,
        metric=args.metric,
        merge_mahalanobis=None if args.no_mahalanobis_merge else 2.0,
        root_merge=not args.no_root_merge,
    )
    if args.k == "auto":
        dim = X.shape[1]
        max_k = int(2890 * (dim**-0.8))
        min_k = int(1942 * (dim**-1.61))
        result = model.fit_auto(X, max_k, min_k, steps=30)
    else:
        result = model.fit(X).result_

    out = df.copy()
    out["xshift_label"] = result.labels
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {result.n_clusters} clusters (k={result.k}, metric={result.metric}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
