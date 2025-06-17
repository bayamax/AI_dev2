#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recommend_followees.py
────────────────────────────────────────
Usage:

$ python recommend_followees.py Alice_123
   -> 上位50件をターミナルへ表示

or import:

from recommend_followees import load_system, predict_followees
device, model, posts, acc_vecs = load_system()
top50 = predict_followees("Alice_123", device, model, acc_vecs)
"""

from pathlib import Path
import sys, time, pickle
import numpy as np
import torch
import torch.nn.functional as F

# ───────── パス設定（train 時と合わせる） ─────────
CODE_DIR = Path(__file__).resolve().parent        # /workspace/code
BASE_DIR = CODE_DIR.parent / 'vast'               # /workspace/vast
OUT_DIR  = BASE_DIR / 'attention_pooling_out'

MODEL_PATH = OUT_DIR / 'attention_pooling_follow_model.pt'
POSTS_NPY  = BASE_DIR / 'aggregated_posting_vectors.npy'
ACC_VEC_NPY = OUT_DIR / 'account_vectors.npy'     # ★キャッシュ

POST_DIM    = 3072
ACCOUNT_DIM = 128
MAX_POSTS   = 50
BATCH = 256                                      # 推論バッチ

# ───────── model 定義（学習スクリプトと同じ） ─────────
import torch.nn as nn

class AttentionPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(POST_DIM, 8, batch_first=True)
        self.ln  = nn.LayerNorm(POST_DIM)
        self.proj = nn.Sequential(
            nn.Linear(POST_DIM, ACCOUNT_DIM*2), nn.GELU(),
            nn.Linear(ACCOUNT_DIM*2, ACCOUNT_DIM))
        self.sc  = nn.Linear(POST_DIM, 1, bias=False)
    def forward(self, x, m):
        x = F.normalize(x, p=2, dim=-1)
        x = self.ln(x + self.mha(x, x, x, key_padding_mask=(m==0))[0])
        w = F.softmax(self.sc(x).squeeze(-1).masked_fill(m==0, -1e9), -1)
        v = (x * w.unsqueeze(-1)).sum(1)
        return F.normalize(self.proj(v), p=2, dim=-1)

class FollowPred(nn.Module):
    def __init__(self):
        super().__init__()
        h = 256
        self.fe = nn.Sequential(nn.Linear(ACCOUNT_DIM, h), nn.GELU())
        self.te = nn.Sequential(nn.Linear(ACCOUNT_DIM, h), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(h*3, h), nn.GELU(),
            nn.Linear(h, h//2), nn.GELU(),
            nn.Linear(h//2, 1), nn.Sigmoid())
    def forward(self, f, t):
        f, t = self.fe(f), self.te(t)
        return self.head(torch.cat([f, t, f*t], -1)).squeeze(-1)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = AttentionPool()
        self.fp = FollowPred()
    def forward(self, fp, fm, tp, tm):
        return self.fp(self.ap(fp,fm), self.ap(tp,tm))

# ───────── 投稿 → Account Vector 変換 ─────────
def pad_posts(lst):
    arr = np.zeros((MAX_POSTS, POST_DIM), np.float32)
    msk = np.zeros(MAX_POSTS, np.float32)
    L = min(len(lst), MAX_POSTS)
    if L:
        arr[:L] = lst[:L]; msk[:L] = 1
    return arr, msk

def build_account_vectors(device, model_ap, posts):
    if ACC_VEC_NPY.exists():
        return np.load(ACC_VEC_NPY, allow_pickle=True).item()

    acc_vecs = {}
    users = list(posts.keys())
    model_ap.eval()
    with torch.no_grad():
        for i in range(0, len(users), BATCH):
            batch_users = users[i:i+BATCH]
            fp, fm = zip(*(pad_posts(posts[u]) for u in batch_users))
            fp, fm = torch.tensor(fp).to(device), torch.tensor(fm).to(device)
            v = model_ap(fp, fm).cpu().numpy()
            acc_vecs.update({u: vec for u, vec in zip(batch_users, v)})
            if (i//BATCH) % 10 == 0:
                print(f"[AP] {i+len(batch_users)}/{len(users)}  done")
    np.save(ACC_VEC_NPY, acc_vecs, allow_pickle=True)
    return acc_vecs

# ───────── システム一括ロード ─────────
def load_system():
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))
    print("device =", device)

    # posts dict
    posts = np.load(POSTS_NPY, allow_pickle=True).item()

    # model
    model = Model().to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # account vectors
    acc_vecs = build_account_vectors(device, model.ap, posts)
    return device, model, posts, acc_vecs

# ───────── 推奨フォロイー予測 ─────────
def predict_followees(user, device, model, acc_vecs, top_k=50):
    if user not in acc_vecs:
        raise ValueError(f"user '{user}' not found in posts data")

    f_vec = torch.tensor(acc_vecs[user]).to(device).unsqueeze(0)  # (1,D)

    tgt_users = [u for u in acc_vecs.keys() if u != user]
    scores = []
    with torch.no_grad():
        for i in range(0, len(tgt_users), BATCH):
            batch_users = tgt_users[i:i+BATCH]
            t_vec = torch.tensor([acc_vecs[u] for u in batch_users]).to(device)
            f_batch = f_vec.repeat(len(batch_users), 1)
            prob = model.fp(f_batch, t_vec).cpu().numpy()  # (B,)
            scores.extend(zip(batch_users, prob))

    top = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    return top

# ───────── CLI ─────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python recommend_followees.py <AccountName> [K]")
        sys.exit(1)
    target = sys.argv[1]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    device, model, posts, acc_vecs = load_system()
    t0 = time.time()
    topK = predict_followees(target, device, model, acc_vecs, K)
    dt = time.time() - t0

    print(f"\nTop {K} followee predictions for '{target}'  ({dt:.2f}s)")
    for rank, (u, p) in enumerate(topK, 1):
        print(f"{rank:>2d}. {u:20s}  prob={p:.3f}")