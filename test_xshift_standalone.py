from pathlib import Path

import numpy as np
import pandas as pd

from xshift_standalone import run_xshift_wrapper


def test_standalone_wrapper_reads_java_style_inputs(tmp_path: Path):
    rng = np.random.default_rng(3)
    data = np.vstack(
        [
            rng.normal([0.2, 0.2, 1.0], 0.03, size=(25, 3)),
            rng.normal([2.5, 0.2, 1.0], 0.03, size=(25, 3)),
        ]
    )
    sample = tmp_path / "sample.csv"
    pd.DataFrame(data, columns=["A", "B", "C"]).to_csv(sample, index=False)

    (tmp_path / "fcsFileList.txt").write_text(str(sample) + "\n")
    (tmp_path / "importConfig.txt").write_text(
        "\n".join(
            [
                "clustering_columns=1,2",
                "side_columns=",
                "rescale=NONE",
                "transformation=NONE",
                "scaling_factor=5",
                "quantile=0.95",
                "rescale_separately=false",
                "limit_events_per_file=0",
                "euclidian_length_threshold=0",
                "noise_threshold=0",
            ]
        )
        + "\n"
    )

    out = tmp_path / "out"
    run_xshift_wrapper(
        "8",
        tmp_path / "importConfig.txt",
        tmp_path / "fcsFileList.txt",
        out,
        write_fcs=False,
    )

    clustered = pd.read_csv(out / "sample_clustered.csv")
    assert "cluster_id" in clustered.columns
    assert clustered["cluster_id"].nunique() >= 2
    assert (out / "mst.gml").exists()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        test_standalone_wrapper_reads_java_style_inputs(Path(td))
