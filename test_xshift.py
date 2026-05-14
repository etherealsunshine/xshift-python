import numpy as np

from xshift import XShift


def test_euclidean_blobs_have_stable_labels():
    rng = np.random.default_rng(1)
    X = np.vstack(
        [
            rng.normal([0, 0], 0.12, size=(40, 2)),
            rng.normal([3, 0], 0.12, size=(40, 2)),
            rng.normal([0, 3], 0.12, size=(40, 2)),
        ]
    )

    model = XShift(k=10, n_size=20, metric="euclidean", root_merge=False, merge_mahalanobis=None)
    labels = model.fit_predict(X)

    assert model.result_.n_clusters == 3
    assert sorted(np.bincount(labels).tolist()) == [40, 40, 40]


def test_angular_profiles_separate_without_root_merge():
    rng = np.random.default_rng(2)
    means = np.array([[1, 4, 1, 1], [4, 1, 1, 1], [1, 1, 4, 1]], dtype=float)
    X = np.vstack([rng.lognormal(np.log(mean), 0.08, size=(35, 4)) for mean in means])

    model = XShift(k=10, n_size=20, metric="angular", root_merge=False, merge_mahalanobis=None)
    labels = model.fit_predict(X)

    assert model.result_.n_clusters == 3
    assert sorted(np.bincount(labels).tolist()) == [35, 35, 35]


def test_root_merge_steps_match_java_double_loop():
    steps = XShift._java_root_merge_steps(0.05)

    assert len(steps) == 18
    assert steps[-1] < 0.95


if __name__ == "__main__":
    test_euclidean_blobs_have_stable_labels()
    test_angular_profiles_separate_without_root_merge()
    test_root_merge_steps_match_java_double_loop()
