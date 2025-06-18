#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_to_accountvector_ranked_fast_val.py
────────────────────────────────────────────────────────
● 投稿 Tensor を fp16 でキャッシュ → 1 epoch ≈ 1 h（Titan X 12 GB）
● AMP + GradScaler で半精度学習
● Hard Neg 80 % + Easy Neg 20 %、HardNeg は 2 epoch ごと再計算
● train loss と valAUC を同時に記録し、最良モデルを自動保存
● rank_follow_model.pt があればロードして続きから再開
"""

from pathlib import Path
import csv, random, time, traceback, math
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ───── DIRS & HYPER ──────────────────────────────────
BASE_DIR   = Path('/workspace/vast')
POSTS_NPY  = BASE_DIR / 'aggregated_posting_vectors.npy'
EDGES_CSV  = BASE_DIR / 'edges.csv'

CACHE_DIR  = BASE_DIR / 'cache' ; CACHE_DIR.mkdir(exist_ok=True)
TRAIN_NPY  = CACHE_DIR / 'train_edges.npy'
VAL_NPY    = CACHE_DIR / 'val_edges.npy'

OUT_DIR    = BASE_DIR / 'attention_rank_out' ; OUT_DIR.mkdir(exist_ok=True)
LOG_FILE   = OUT_DIR / 'rank_training.log'
CKPT_PATH  = OUT_DIR / 'rank_follow_model.pt'

POST_DIM, ACCOUNT_DIM, HIDDEN_DIM = 3072, 128, 256
MAX_POSTS, BATCH_SIZE             = 50, 128
NUM_EPOCHS, LR, WD, EARLY_STOP    = 50, 3e-4, 1e-5, 8

NEG_PER_POS   = 3
HARD_TOPN     = 20
EASY_NEG_RATE = 0.2
VALID_RATIO   = 0.15
HN_INTERVAL   = 2
NUM_WORKERS   = 4
FP16          = True                           # 端末の VRAM に応じて False でも可
# ────────────────────────────────────────────────

def log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}  {msg}")
    with LOG_FILE.open('a') as f: f.write(f"{ts}  {msg}\n")

# ---- load / split data ------------------------------------------------
def load_posts():
    return np.load(POSTS_NPY, allow_pickle=True).item()

def load_edges():
    with EDGES_CSV.open() as f:
        rdr = csv.reader(f); next(rdr)
        return [tuple(r[:2]) for r in rdr]

def split_edges(edges):
    random.shuffle(edges)
    k = int(len(edges) * (1 - VALID_RATIO))
    np.save(TRAIN_NPY, np.array(edges[:k], dtype=object), allow_pickle=True)
    np.save(VAL_NPY,   np.array(edges[k:],  dtype=object), allow_pickle=True)
    return edges[:k], edges[k:]

# ---- model ------------------------------------------------------------
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
    def score(self, fp, fm, tp, tm):
        return self.fp(self.ap(fp, fm), self.ap(tp, tm))

# ---- util -------------------------------------------------------------
def pad_posts(lst):
    arr = np.zeros((MAX_POSTS, POST_DIM), np.float32)
    msk = np.zeros(MAX_POSTS, np.float32)
    L = min(len(lst), MAX_POSTS)
    if L: arr[:L] = lst[:L]; msk[:L] = 1
    return arr, msk

def build_post_cache(posts, device):
    cache = {}
    dtype = torch.float16 if FP16 else torch.float32
    for u, vecs in posts.items():
        fp, fm = pad_posts(vecs)
        cache[u] = (torch.tensor(fp, dtype=dtype, device=device),
                    torch.tensor(fm, dtype=dtype, device=device))
    return cache

def compute_acc_vecs(cache, device, ap):
    users = list(cache.keys())
    dtype = torch.float16 if FP16 else torch.float32
    V = torch.zeros((len(users), ACCOUNT_DIM), dtype=dtype, device=device)
    ap.eval()
    with torch.no_grad():
        for i in range(0, len(users), 512):
            fp, fm = zip(*(cache[u] for u in users[i:i+512]))
            fp = torch.stack(fp); fm = torch.stack(fm)
            V[i:i + len(fp)] = ap(fp, fm)
    return users, V

def hard_negs(users, V, pos_edges):
    idx = {u: i for i, u in enumerate(users)}
    edge_set = set(pos_edges)
    sims = F.normalize(V, dim=1) @ F.normalize(V, dim=1).T
    neg = []
    for f, _ in pos_edges:
        if f not in idx: continue
        i = idx[f]
        _, ids = sims[i].topk(HARD_TOPN + 1)
        cands = [users[j] for j in ids[1:].tolist() if (f, users[j]) not in edge_set]
        random.shuffle(cands)
        neg.extend([(f, u) for u in cands[:NEG_PER_POS]])
    return neg

def easy_negs(valid_users, pos_edges):
    need = int(len(pos_edges) * EASY_NEG_RATE)
    edge_set = set(pos_edges); neg = []
    while len(neg) < need:
        f, t = random.sample(valid_users, 2)
        if (f, t) not in edge_set: neg.append((f, t))
    return neg

# ---- Dataset ----------------------------------------------------------
class TripletDS(Dataset):
    def __init__(self, pos, neg, cache):
        self.cache = cache
        neg_by_f = defaultdict(list)
        for f, n in neg: neg_by_f[f].append(n)
        self.trip = [(f, p, n) for f, p in pos for n in neg_by_f[f]]
    def __len__(self): return len(self.trip)
    def __getitem__(self, i):
        f, p, n = self.trip[i]
        fp, fm = self.cache[f]
        tp, tm = self.cache[p]
        np_, nm = self.cache[n]
        return fp, fm, tp, tm, np_, nm

# ---- Validation -------------------------------------------------------
def build_val_dl(val_edges, cache):
    users = list(cache.keys())
    neg = [(f, random.choice(users)) for f, _ in val_edges]
    ds = TripletDS(val_edges, neg, cache)
    return DataLoader(ds, batch_size=512, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)

# ---- train loop -------------------------------------------------------
def train():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"device={dev}")

    posts = load_posts()
    edges_all = load_edges()
    valid_users = {u for u, p in posts.items() if p}
    edges = [(u, v) for u, v in edges_all if u in valid_users and v in valid_users]
    train_e, val_e = split_edges(edges)

    post_cache = build_post_cache(posts, dev)
    val_dl = build_val_dl(val_e, post_cache)

    model = Model().to(dev)
    if CKPT_PATH.exists():
        model.load_state_dict(torch.load(CKPT_PATH, map_location=dev))
        log("checkpoint loaded → 再開")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler = torch.cuda.amp.GradScaler(enabled=FP16)

    best_auc, wait = 0.0, 0
    for ep in range(1, NUM_EPOCHS + 1):
        # HardNeg 更新タイミング
        if ep == 1 or ep % HN_INTERVAL == 0:
            users, V = compute_acc_vecs(post_cache, dev, model.ap)
            hard = hard_negs(users, V, train_e)
        easy = easy_negs(list(valid_users), train_e)

        ds = TripletDS(train_e, hard + easy, post_cache)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True, num_workers=NUM_WORKERS,
                        pin_memory=True, persistent_workers=True)

        # ---- train ----
        model.train(); tot = 0
        for fp, fm, tp, tm, np_, nm in tqdm(dl, desc=f"E{ep}"):
            fp, fm, tp, tm, np_, nm = [x.to(dev) for x in (fp, fm, tp, tm, np_, nm)]
            with torch.cuda.amp.autocast(enabled=FP16):
                pos = model.score(fp, fm, tp, tm)
                neg = model.score(fp, fm, np_, nm)
                loss = -torch.log(torch.sigmoid(pos - neg)).mean()
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item() * len(pos)
        train_loss = tot / len(ds)

        # ---- validation ----
        model.eval(); hit = total = 0
        with torch.no_grad():
            for fp, fm, tp, tm, np_, nm in val_dl:
                fp, fm, tp, tm, np_, nm = [x.to(dev) for x in (fp, fm, tp, tm, np_, nm)]
                pos = model.score(fp, fm, tp, tm).sigmoid()
                neg = model.score(fp, fm, np_, nm).sigmoid()
                hit += (pos > neg).sum().item()
                total += len(pos)
        val_auc = hit / total
        log(f"[{ep}] train {train_loss:.4f} | valAUC {val_auc:.3f}")

        # ---- checkpoint ----
        if val_auc > best_auc:
            best_auc, wait = val_auc, 0
            torch.save(model.state_dict(), CKPT_PATH)
            log("  saved best checkpoint")
        else:
            wait += 1
            if wait >= EARLY_STOP:
                log("early stop"); break

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        log(f"Error: {e}\n{traceback.format_exc()}")