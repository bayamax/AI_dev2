#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recommend_followees_pop.py
────────────────────────────────────────────────────────
USAGE:
    python recommend_followees_pop.py --user <account> [--topk 20] [--alpha 0.5]

    score_final = score_model * (1 / (1+deg)) ** alpha
      alpha = 0   → 補正なし (元のモデルスコア)
      alpha > 0   → 有名アカを弱め、マイナーをブースト
────────────────────────────────────────────────────────
"""

from pathlib import Path
import argparse, time, csv, math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ───── DIR / CONST (学習コードと合わせる) ─────────────────────
BASE_DIR   = Path('/workspace/workspace/vast')          # 適宜変更
POSTS_NPY  = BASE_DIR / 'aggregated_posting_vectors.npy'
EDGES_CSV  = BASE_DIR / 'edges.csv'
CKPT_PATH  = BASE_DIR / 'attention_rank_out' / 'rank_follow_model.pt'

POST_DIM, ACCOUNT_DIM, HIDDEN_DIM = 3072, 128, 256
MAX_POSTS                        = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ───── Model (学習時と同一) ──────────────────────────────────
class AttentionPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha  = nn.MultiheadAttention(POST_DIM, 8, batch_first=True)
        self.ln   = nn.LayerNorm(POST_DIM)
        self.proj = nn.Sequential(
            nn.Linear(POST_DIM, ACCOUNT_DIM * 2), nn.GELU(),
            nn.Linear(ACCOUNT_DIM * 2, ACCOUNT_DIM))
        self.sc   = nn.Linear(POST_DIM, 1, bias=False)

    def forward(self, x, m):
        x = x.float()
        x = F.normalize(x, p=2, dim=-1)
        x = self.ln(x + self.mha(x, x, x, key_padding_mask=(m == 0))[0])
        w = F.softmax(self.sc(x).squeeze(-1).masked_fill(m == 0, -1e9), -1)
        v = (x * w.unsqueeze(-1)).sum(1)
        return F.normalize(self.proj(v), p=2, dim=-1)

class FollowPred(nn.Module):
    def __init__(self):
        super().__init__(); h = HIDDEN_DIM
        self.fe = nn.Sequential(nn.Linear(ACCOUNT_DIM, h), nn.GELU())
        self.te = nn.Sequential(nn.Linear(ACCOUNT_DIM, h), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(h * 3, h), nn.GELU(),
            nn.Linear(h, h // 2), nn.GELU(),
            nn.Linear(h // 2, 1))

    def forward(self, f, t):
        f, t = self.fe(f), self.te(t)
        return self.head(torch.cat([f, t, f * t], -1)).squeeze(-1)

class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.ap = AttentionPool(); self.fp = FollowPred()
    def encode(self, p, m):           # 投稿集合→128次元
        return self.ap(p, m)
    def score_vec(self, f_vec, t_vec):
        return self.fp(f_vec, t_vec)

# ───── Utility --------------------------------------------------------
def pad_posts(lst):
    arr = np.zeros((MAX_POSTS, POST_DIM), np.float32)
    msk = np.zeros(MAX_POSTS, np.float32)
    L = min(len(lst), MAX_POSTS)
    if L:
        arr[:L] = lst[:L]
        msk[:L] = 1
    return arr, msk

def build_post_cache(posts):
    cache = {}
    for u, vecs in posts.items():
        fp, fm = pad_posts(vecs)
        cache[u] = (torch.tensor(fp, device='cpu'),
                    torch.tensor(fm, device='cpu'))
    return cache

def compute_all_vectors(model, cache):
    users = list(cache.keys())
    V = torch.zeros((len(users), ACCOUNT_DIM), device=DEVICE)
    model.ap = model.ap.to(DEVICE).eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(users), 512), desc="Encoding users"):
            fp, fm = zip(*(cache[u] for u in users[i:i+512]))
            fp = torch.stack(fp).to(DEVICE).float()
            fm = torch.stack(fm).to(DEVICE).float()
            V[i:i+len(fp)] = model.ap(fp, fm)
    return users, V

def load_edges():
    with EDGES_CSV.open() as f:
        rdr = csv.reader(f); next(rdr)
        return [tuple(r[:2]) for r in rdr]

# ───── Main 推薦ルーチン ----------------------------------------------
def recommend(user, topk, alpha):
    # データ読み込み
    posts = np.load(POSTS_NPY, allow_pickle=True).item()
    if user not in posts or len(posts[user]) == 0:
        raise ValueError(f"User '{user}' not found or no posts.")
    cache = build_post_cache(posts)

    # 人気度 (in-degree) 計算
    in_deg = {}
    for _, tgt in load_edges():
        in_deg[tgt] = in_deg.get(tgt, 0) + 1
    mean_deg = np.mean(list(in_deg.values())) if in_deg else 1.0

    # モデルロード
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()

    # 全ユーザのアカウントベクトル
    users, V = compute_all_vectors(model, cache)
    idx = {u: i for i, u in enumerate(users)}

    # フォロワー側ベクトル
    with torch.no_grad():
        fp, fm = cache[user]
        f_vec = model.ap(fp.unsqueeze(0).to(DEVICE).float(),
                         fm.unsqueeze(0).to(DEVICE).float())

    # スコア計算
    with torch.no_grad():
        base_scores = model.fp(f_vec.repeat(len(V), 1), V).cpu().numpy()

    # 人気度補正
    weights = np.array([(mean_deg + 1) / (in_deg.get(u, 0) + 1) for u in users])
    #  (mean_deg+1)/(deg+1) なので平均≈1、deg↓ ⇒ weight↑（下駄）
    weights = weights ** alpha          # alpha=0 → 補正なし
    final_scores = base_scores * weights

    # 既フォロー排除
    edge_set = set(load_edges())
    already  = {v for u, v in edge_set if u == user} | {user}
    cand_idx = [i for i, u in enumerate(users) if u not in already]

    # Top-K
    cand = [(users[i], float(base_scores[i]), float(weights[i]), float(final_scores[i]))
            for i in cand_idx]
    cand.sort(key=lambda x: x[3], reverse=True)
    return cand[:topk]

# ───── CLI ------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--user", required=True, help="follower account name")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="0→補正なし, 大きいほどマイナー優遇")
    args = p.parse_args()

    t0 = time.time()
    recs = recommend(args.user, args.topk, args.alpha)
    print(f"\n=== Recommendations for '{args.user}'  (alpha={args.alpha}) ===")
    print("rank | account               | base  | wght | final")
    print("-----+-----------------------+-------+------+-------")
    for r, (u, b, w, f) in enumerate(recs, 1):
        print(f"{r:3d}  | {u:22s} | {b:+.4f} | {w:.3f} | {f:+.4f}")
    print(f"-------------------------------------------------------------")
    print(f"Done in {time.time()-t0:.1f}s  (device={DEVICE})")
