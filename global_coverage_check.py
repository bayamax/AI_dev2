#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
global_coverage_check.py
────────────────────────────────────────────────────────
1. すべての投稿ベクトルをロード      (aggregated_posting_vectors.npy)
2. 基本統計
      • 総ベクトル数・平均ノルム
      • 全体平均ベクトル ∥μ∥
3. ペアワイズ余弦類似ヒストグラム (ランダム10万ペア)
      -> mean / std が 0 ±0.1 付近なら球殻っぽい
4. 主成分解析（上位20次元）
      • 各成分の分散寄与率
      -> 上位が15〜25%程度に集中 → 低ランク構造は弱め
────────────────────────────────────────────────────────
USAGE:
    python global_coverage_check.py
"""

from pathlib import Path
import random, time
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

# ---------- パス設定 ----------------------------------------------------
BASE_DIR  = Path('/workspace/workspace/vast')  # 適宜変更
POSTS_NPY = BASE_DIR / 'aggregated_posting_vectors.npy'
PAIR_SAMPLES = 100_000       # 余弦類似をランダムに何ペア取るか

# ---------- 1. 投稿ベクトル読み込み ------------------------------------
t0 = time.time()
post_obj = np.load(POSTS_NPY, allow_pickle=True).item()  # {user: [vecs]}
vecs = np.vstack([v for vlist in post_obj.values() for v in vlist])
N, D = vecs.shape
print(f"[load] vectors = {N:,}   dim = {D}   time = {time.time()-t0:.1f}s")

# ---------- 2. 基本統計 -------------------------------------------------
norms = np.linalg.norm(vecs, axis=1)
print(f"\n=== Basic stats ===")
print(f"mean‖x‖   : {norms.mean():.4f}")
print(f"std ‖x‖   : {norms.std():.4f}")
mu = vecs.mean(axis=0)
print(f"‖overall mean‖ : {np.linalg.norm(mu):.6f} (should be ≪ 1 if isotropic)")

# ---------- 3. 余弦類似ヒストグラム -------------------------------------
print(f"\n=== Cosine-sim distribution (random {PAIR_SAMPLES:,} pairs) ===")
idx = np.random.choice(N, size=(PAIR_SAMPLES, 2), replace=True)
a = vecs[idx[:,0]]
b = vecs[idx[:,1]]
cos = np.sum(a*b, axis=1) / (np.linalg.norm(a,1)*np.linalg.norm(b,1))
print(f"mean : {cos.mean():.4f}")
print(f"std  : {cos.std():.4f}")
hist, bin_edges = np.histogram(cos, bins=30, range=(-0.2,0.8))
for h, lo, hi in zip(hist, bin_edges[:-1], bin_edges[1:]):
    bar = "█"*int(h/max(hist)*40)
    print(f"{lo:+.2f}–{hi:+.2f} | {bar}")

# ---------- 4. PCA 上位20成分 ------------------------------------------
print(f"\n=== PCA (top 20 components) ===")
pca = PCA(n_components=20, svd_solver='randomized', random_state=42)
pca.fit(vecs[np.random.choice(N, min(100_000, N), replace=False)])
var = pca.explained_variance_ratio_
for i,v in enumerate(var,1):
    print(f"PC{i:2d}: {v*100:5.2f}%")
print(f"Σ top20 : {var.sum()*100:5.2f}%  (total variance explained)")

print("\nDone.")