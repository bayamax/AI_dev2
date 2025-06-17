#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_to_accountvector_ranked.py
────────────────────────────────────────────────────────
Hard Negative 80 % + Easy Negative 20 % で BPR ランキング学習
・投稿が 1 件も無いユーザーは正例・負例とも完全除外
・list/tuple 混在バグ解消
・tensor 警告解消（np.stack → tensor）
"""

from pathlib import Path
import csv, random, traceback, time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ───── ディレクトリ & ハイパーパラメータ ─────────────────
BASE_DIR   = Path('/workspace/vast')
POSTS_NPY  = BASE_DIR / 'aggregated_posting_vectors.npy'
EDGES_CSV  = BASE_DIR / 'edges.csv'

CACHE_DIR  = BASE_DIR / 'cache'; CACHE_DIR.mkdir(exist_ok=True)
TRAIN_NPY  = CACHE_DIR / 'train_edges.npy'
VAL_NPY    = CACHE_DIR / 'val_edges.npy'

OUT_DIR    = BASE_DIR / 'attention_rank_out'; OUT_DIR.mkdir(exist_ok=True)
LOG_FILE   = OUT_DIR / 'rank_training.log'
MODEL_PATH = OUT_DIR / 'rank_follow_model.pt'

POST_DIM, ACCOUNT_DIM, HIDDEN_DIM = 3072, 128, 256
MAX_POSTS, BATCH_SIZE             = 50, 256
NUM_EPOCHS, LR, WD, EARLY_STOP    = 50, 3e-4, 1e-5, 8

NEG_PER_POS   = 3         # HardNeg / 正例
HARD_TOPN     = 20        # HardNeg 候補は類似度上位 20
EASY_NEG_RATE = 0.2       # EasyNeg : 正例比 20 %
VALID_RATIO   = 0.15
# ───────────────────────────────────────────────────────


def log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}  {msg}")
    with LOG_FILE.open('a') as f:
        f.write(f"{ts}  {msg}\n")


# ---- データ読み込み --------------------------------------------------
def load_posts():
    return np.load(POSTS_NPY, allow_pickle=True).item()      # {uid: [vec,…]}


def load_edges():
    with EDGES_CSV.open() as f:
        rdr = csv.reader(f)
        next(rdr)
        return [tuple(r[:2]) for r in rdr]                   # [(u,v), …]


def split_edges(edges):
    random.shuffle(edges)
    k = int(len(edges) * (1 - VALID_RATIO))
    np.save(TRAIN_NPY, np.array(edges[:k], dtype=object), allow_pickle=True)
    np.save(VAL_NPY,   np.array(edges[k:],  dtype=object), allow_pickle=True)
    return edges[:k], edges[k:]


# ---- モデル ----------------------------------------------------------
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
        x = F.normalize(x, p=2, dim=-1)
        x = self.ln(x + self.mha(x, x, x, key_padding_mask=(m == 0))[0])
        w = F.softmax(
            self.sc(x).squeeze(-1).masked_fill(m == 0, -1e9), -1)
        v = (x * w.unsqueeze(-1)).sum(1)
        return F.normalize(self.proj(v), p=2, dim=-1)


class FollowPred(nn.Module):
    def __init__(self):
        super().__init__()
        h = HIDDEN_DIM
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
        super().__init__()
        self.ap = AttentionPool()
        self.fp = FollowPred()

    def score(self, fp, fm, tp, tm):
        return self.fp(self.ap(fp, fm), self.ap(tp, tm))


# ---- ユーティリティ --------------------------------------------------
def pad_posts(lst):
    arr = np.zeros((MAX_POSTS, POST_DIM), np.float32)
    msk = np.zeros(MAX_POSTS, np.float32)
    L = min(len(lst), MAX_POSTS)
    if L:
        arr[:L] = lst[:L]
        msk[:L] = 1
    return arr, msk


def compute_acc_vecs(posts, dev, ap):
    users = list(posts.keys())
    # 正しい形 (U, ACCOUNT_DIM) で初期化
    V = np.zeros((len(users), ACCOUNT_DIM), dtype=np.float32)

    ap.eval()
    with torch.no_grad():
        for i in range(0, len(users), 512):
            fp, fm = zip(*(pad_posts(posts[u])
                           for u in users[i:i + 512]))
            fp = torch.tensor(np.array(fp), device=dev)
            fm = torch.tensor(np.array(fm), device=dev)
            v = ap(fp, fm).cpu().numpy()
            V[i:i + len(v)] = v
    return users, V


def hard_negs(users, V, pos_edges):
    idx = {u: i for i, u in enumerate(users)}
    edge_set = set(pos_edges)
    M = torch.tensor(V)
    M = F.normalize(M, dim=1)
    sims = M @ M.T
    neg = []
    for f, _ in pos_edges:
        if f not in idx:
            continue
        i = idx[f]
        _, ids = sims[i].topk(HARD_TOPN + 1)
        cands = [users[j] for j in ids[1:].tolist()
                 if (f, users[j]) not in edge_set]
        random.shuffle(cands)
        neg.extend([(f, u) for u in cands[:NEG_PER_POS]])
    return neg


def easy_negs(valid_users, pos_edges):
    need = int(len(pos_edges) * EASY_NEG_RATE)
    edge_set = set(pos_edges)
    neg = []
    while len(neg) < need:
        f, t = random.sample(valid_users, 2)
        if (f, t) not in edge_set:
            neg.append((f, t))
    return neg


# ---- Dataset ---------------------------------------------------------
class TripletDS(Dataset):
    def __init__(self, pos, neg, posts):
        self.posts = posts
        self.trip = []
        neg_by_f = defaultdict(list)
        for f, n in neg:
            neg_by_f[f].append(n)
        for f, p in pos:
            for n in neg_by_f[f]:
                self.trip.append((f, p, n))

    def __len__(self):
        return len(self.trip)

    def __getitem__(self, i):
        f, p, n = self.trip[i]
        fp, fm = pad_posts(self.posts[f])
        tp, tm = pad_posts(self.posts[p])
        np_, nm = pad_posts(self.posts[n])
        return map(torch.tensor, (fp, fm, tp, tm, np_, nm))


# ---- メインループ -----------------------------------------------------
def train():
    dev = (torch.device('cuda') if torch.cuda.is_available()
           else torch.device('cpu'))
    log(f"device={dev}")

    posts = load_posts()
    edges_all = load_edges()

    # 投稿があるユーザーのみ残す
    valid_users = {u for u, p in posts.items() if p}
    edges = [(u, v) for u, v in edges_all
             if u in valid_users and v in valid_users]

    train_e, val_e = split_edges(edges)

    model = Model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best, wait = 1e9, 0
    for ep in range(1, NUM_EPOCHS + 1):
        users, V = compute_acc_vecs(posts, dev, model.ap)
        hard = hard_negs(users, V, train_e)
        easy = easy_negs(list(valid_users), train_e)
        ds = TripletDS(train_e, hard + easy, posts)
        dl = DataLoader(ds, batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True)

        model.train()
        tot = 0
        for batch in tqdm(dl, desc=f"E{ep}"):
            fp, fm, tp, tm, np_, nm = [x.to(dev) for x in batch]
            pos = model.score(fp, fm, tp, tm)
            neg = model.score(fp, fm, np_, nm)
            loss = -torch.log(torch.sigmoid(pos - neg)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(pos)
        avg = tot / len(ds)
        log(f"[{ep}] loss={avg:.4f}")

        if avg < best:
            best, wait = avg, 0
            torch.save(model.state_dict(), MODEL_PATH)
            log("  saved best")
        else:
            wait += 1
            if wait >= EARLY_STOP:
                log("early stop")
                break


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        log(f"Error: {e}\n{traceback.format_exc()}")