#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recommend_followees.py
────────────────────────────────────────────────────────
USAGE:
    python recommend_followees.py --user <account_name> [--topk 20]
--------------------------------------------------------
1. 事前学習済み ckpt (rank_follow_model.pt) をロード
2. 全ユーザのアカウントベクトルを一括計算＆キャッシュ
3. 指定ユーザの followee スコアを算出し、Top-K を出力
   （既にフォロー中のアカウントは自動で除外）
"""

from pathlib import Path
import argparse, time, csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ───── DIRS & CONST (学習コードと合わせる) ─────────────────────────
BASE_DIR   = Path('/workspace/workspace/vast')
POSTS_NPY  = BASE_DIR / 'aggregated_posting_vectors.npy'
EDGES_CSV  = BASE_DIR / 'edges.csv'
CKPT_PATH  = BASE_DIR / 'attention_rank_out' / 'rank_follow_model.pt'

POST_DIM, ACCOUNT_DIM, HIDDEN_DIM = 3072, 128, 256
MAX_POSTS                        = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ───── Model 定義 (学習時と同一) ────────────────────────────────
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
    def score_vec(self, f_vec, t_vec):# 128×128 → スコア
        return self.fp(f_vec, t_vec)

# ───── Utility ─────────────────────────────────────────────────
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
    """AttentionPool を GPU に乗せて一括でアカウントベクトルを計算"""
    users = list(cache.keys())
    V = torch.zeros((len(users), ACCOUNT_DIM), device=DEVICE)
    model.ap = model.ap.to(DEVICE).eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(users), 512), desc="Encoding users"):
            fp, fm = zip(*(cache[u] for u in users[i:i+512]))
            fp = torch.stack(fp).to(DEVICE).float()
            fm = torch.stack(fm).to(DEVICE).float()
            V[i:i+len(fp)] = model.ap(fp, fm)
    return users, V               # users[i] ↔ V[i]

def load_edges():
    with EDGES_CSV.open() as f:
        rdr = csv.reader(f); next(rdr)
        return [tuple(r[:2]) for r in rdr]

# ───── Main 推薦ルーチン ────────────────────────────────────────
def recommend(user, topk):
    # ① データ読み込み
    posts = np.load(POSTS_NPY, allow_pickle=True).item()
    if user not in posts or len(posts[user]) == 0:
        raise ValueError(f"User '{user}' has no posts or does not exist.")
    cache = build_post_cache(posts)

    # ② モデルロード
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()

    # ③ 全ユーザのアカウントベクトルを計算
    users, V = compute_all_vectors(model, cache)
    idx = {u: i for i, u in enumerate(users)}

    # ④ 指定ユーザのベクトル取得
    with torch.no_grad():
        fp, fm = cache[user]
        f_vec  = model.ap(fp.unsqueeze(0).to(DEVICE).float(),
                          fm.unsqueeze(0).to(DEVICE).float())  # (1,128)

    # ⑤ スコア計算（行列演算で高速）
    with torch.no_grad():
        scores = model.fp(f_vec.repeat(len(V), 1), V).cpu().numpy()  # (N,)

    # ⑥ 既フォローは除外
    edge_set = set(load_edges())
    already  = {v for u, v in edge_set if u == user} | {user}
    cand_idx = [i for i, u in enumerate(users) if u not in already]

    # ⑦ Top-K 取得
    cand_scores = [(users[i], float(scores[i])) for i in cand_idx]
    cand_scores.sort(key=lambda x: x[1], reverse=True)
    return cand_scores[:topk]

# ───── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True, help="follower account name")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    t0 = time.time()
    recs = recommend(args.user, args.topk)
    dt = time.time() - t0

    print(f"\n=== Recommendations for '{args.user}' (top {args.topk}) ===")
    for rank, (u, s) in enumerate(recs, 1):
        print(f"{rank:2d}. {u:20s} | score {s:+.4f}")
    print(f"---------------------------------------------------")
    print(f"Done in {dt:.1f} sec  (device={DEVICE})")