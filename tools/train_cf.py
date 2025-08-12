"""
Very small MF training on synthetic or provided interactions to produce a demo CF model.

Usage:
  python -m tools.train_cf --data examples/interactions.jsonl --k 16 --it 10

If --data is missing, a tiny synthetic dataset will be generated.
Saves model to backend/app/ml/models/cf.npz (compatible with app.services.cf.CFModel).
"""

import os, json, argparse
import mlflow
import numpy as np


def load_interactions(path: str | None) -> list[dict]:
    if path and os.path.exists(path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        return rows
    # synthetic
    rows = []
    users = [1, 2, 3]
    items = [1, 2, 3, 4, 5]
    rng = np.random.default_rng(7)
    for u in users:
        for q in items:
            rows.append({"user_id": u, "question_id": q, "label": bool(rng.random() > 0.4)})
    return rows


def build_maps(rows: list[dict]) -> tuple[dict[int, int], dict[int, int]]:
    users = sorted({int(r["user_id"]) for r in rows})
    items = sorted({int(r["question_id"]) for r in rows})
    umap = {u: i for i, u in enumerate(users)}
    imap = {q: i for i, q in enumerate(items)}
    return umap, imap


def als(rows: list[dict], umap: dict[int, int], imap: dict[int, int], k: int = 16, lam: float = 0.1, it: int = 10):
    U = np.random.randn(len(umap), k) * 0.1
    V = np.random.randn(len(imap), k) * 0.1
    R = np.zeros((len(umap), len(imap)), dtype=np.float32)
    for r in rows:
        R[umap[int(r["user_id"])], imap[int(r["question_id"])]] = float(1.0 if r.get("label") else 0.0)
    eye = np.eye(k)
    for _ in range(it):
        VV = V.T @ V + lam * eye
        for u in range(U.shape[0]):
            U[u] = np.linalg.solve(VV, V.T @ R[u])
        UU = U.T @ U + lam * eye
        for i in range(V.shape[0]):
            V[i] = np.linalg.solve(UU, U.T @ R[:, i])
    return U, V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="Path to interactions jsonl: {user_id, question_id, label}")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--it", type=int, default=10)
    args = ap.parse_args()

    rows = load_interactions(args.data)
    if not rows:
        print("No interactions; abort")
        return
    umap, imap = build_maps(rows)
    with mlflow.start_run(run_name="train_cf"):
        mlflow.log_params({"k": args.k, "it": args.it})
        U, V = als(rows, umap, imap, k=args.k, it=args.it)
        mlflow.log_metric("users", len(umap))
        mlflow.log_metric("items", len(imap))
    out_dir = "backend/app/ml/models"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cf.npz")
        np.savez(out_path, U=U, V=V, user_map=umap, item_map=imap)
        mlflow.log_artifact(out_path)
        print(f"Saved CF model to {out_path} (users={len(umap)}, items={len(imap)}, k={args.k})")


if __name__ == "__main__":
    main()


