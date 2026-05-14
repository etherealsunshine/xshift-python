from multiprocessing import Process, Queue
import time


def worker(q):
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    from xshift import XShift

    df = pd.read_csv("data/samusik/samusik01_type_markers.csv")
    X = df.drop(columns=["population_id"]).to_numpy(float)
    truth = df["population_id"].astype(str)
    t0 = time.time()
    model = XShift(k=20, n_size=20, metric="angular", root_merge=False, merge_mahalanobis=None)
    labels = model.fit_predict(X)
    q.put(
        {
            "seconds": time.time() - t0,
            "n_clusters": model.result_.n_clusters,
            "ari": adjusted_rand_score(truth, labels),
            "nmi": normalized_mutual_info_score(truth, labels),
        }
    )


if __name__ == "__main__":
    q = Queue()
    p = Process(target=worker, args=(q,))
    p.start()
    p.join(180)
    if p.is_alive():
        p.terminate()
        p.join()
        print("TIMEOUT after 180 seconds")
    else:
        print(q.get() if not q.empty() else "finished without result")
