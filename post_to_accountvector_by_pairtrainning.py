#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_to_accountvector_by_pairtrainning.py
────────────────────────────────────────────────────────
ディレクトリ構成（固定）:

/workspace/
├─ vast/         ← データと出力をまとめるフォルダ
│   ├─ aggregated_posting_vectors.npy / .csv
│   ├─ edges.csv
│   ├─ cache/                       （自動生成）
│   └─ attention_pooling_out/       （自動生成）
└─ code/         ← このスクリプトを置く場所

※ /workspace/code で
   $ python post_to_accountvector_by_pairtrainning.py
   と実行すれば OK。
"""

# =============== 0. パス & ハイパーパラメータ ===============
from pathlib import Path
import os, csv, time, random, traceback
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─── ディレクトリ設定 ──────────────────────────────
CODE_DIR = Path(__file__).resolve().parent          # /workspace/code
BASE_DIR = CODE_DIR.parent / 'vast'                 # /workspace/vast

POSTS_CSV  = BASE_DIR / 'aggregated_posting_vectors.csv'
POSTS_NPY  = BASE_DIR / 'aggregated_posting_vectors.npy'
EDGES_CSV  = BASE_DIR / 'edges.csv'

CACHE_DIR  = BASE_DIR / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
TRAIN_EDGES_NPY = CACHE_DIR / 'train_edges.npy'
VAL_EDGES_NPY   = CACHE_DIR / 'val_edges.npy'

OUT_DIR   = BASE_DIR / 'attention_pooling_out'
OUT_DIR.mkdir(exist_ok=True)

LOG_FILE   = OUT_DIR / 'attention_pooling_training.log'
MODEL_PATH = OUT_DIR / 'attention_pooling_follow_model.pt'

POST_DIM    = 3072
ACCOUNT_DIM = 128
HIDDEN_DIM  = 256
MAX_POSTS   = 50

NEG_RATIO        = 1        # 既存ユーザー負例
RANDOM_NEG_RATIO = 1        # ランダムベクトル負例
VALID_RATIO      = 0.15

BATCH_SIZE  = 32
NUM_EPOCHS  = 100
LR          = 1e-4
WD          = 1e-5
EARLY_STOP  = 10
FREQ_ALPHA  = 0.5            # 出現回数補正係数
# ============================================================


# =============== 1. ユーティリティ ===============
def log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}  {msg}")
    with LOG_FILE.open('a', encoding='utf-8') as f:
        f.write(f"{ts}  {msg}\n")


# --------------- 投稿読み込み ----------------
def parse_vec(txt: str):
    txt = txt.strip().lstrip('[').rstrip(']').strip('"')
    return None if not txt else np.fromstring(txt, sep=',', dtype=np.float32)


def load_posts() -> dict[str, list[np.ndarray]]:
    if POSTS_NPY.exists():
        log(f"Loading posts cache {POSTS_NPY}")
        return np.load(POSTS_NPY, allow_pickle=True).item()

    if not POSTS_CSV.exists():
        raise FileNotFoundError(
            "posts CSV も .npy キャッシュも見つかりません。\n"
            f"Expected at: {POSTS_CSV} or {POSTS_NPY}"
        )

    log(f"Parsing posts CSV {POSTS_CSV}")
    posts = defaultdict(list)
    with POSTS_CSV.open() as f:
        rdr = csv.reader(f)
        next(rdr)  # ヘッダスキップ
        for uid, _, vstr in tqdm(rdr, desc="posts csv"):
            vec = parse_vec(vstr)
            if vec is not None and len(posts[uid]) < MAX_POSTS:
                posts[uid].append(vec / np.linalg.norm(vec))
    posts = dict(posts)
    np.save(POSTS_NPY, posts, allow_pickle=True)
    log(f"posts users={len(posts)} cached")
    return posts


# --------------- edges split -----------------
def load_split_edges():
    if TRAIN_EDGES_NPY.exists():
        return (
            np.load(TRAIN_EDGES_NPY, allow_pickle=True).tolist(),
            np.load(VAL_EDGES_NPY, allow_pickle=True).tolist()
        )

    log("Splitting edges CSV")
    edges = []
    with EDGES_CSV.open() as f:
        rdr = csv.reader(f)
        next(rdr)
        for src, tgt, *_ in rdr:
            edges.append((src, tgt))

    random.shuffle(edges)
    split = int(len(edges) * (1 - VALID_RATIO))
    np.save(TRAIN_EDGES_NPY, edges[:split], allow_pickle=True)
    np.save(VAL_EDGES_NPY, edges[split:], allow_pickle=True)
    return edges[:split], edges[split:]


# --------------- ネガティブサンプラ -------------
def simple_negative_sampling(users: list[str], k_total: int):
    neg = []
    for _ in range(k_total):
        f = random.choice(users)
        t = random.choice(users)
        while t == f:
            t = random.choice(users)
        neg.append((f, t))
    return neg


# --------------- Dataset ----------------------
class SimplePairDS(Dataset):
    def __init__(
        self,
        pos_edges,
        neg_edges,
        posts,
        pos_freq_dict,
        avg_freq,
        alpha=FREQ_ALPHA,
    ):
        self.posts = posts
        self.alpha = alpha
        self.avg_f = avg_freq
        self.pos_f = pos_freq_dict

        valid = {u for u, p in posts.items() if p}
        self.data = []
        for f, t in pos_edges:
            if f in valid and t in valid:
                self.data.append((f, t, 1.0))
        for f, t in neg_edges:
            if f in valid and (t in valid or t == "__RANDOM_USER__"):
                self.data.append((f, t, 0.0))
        random.shuffle(self.data)

    # --- util ---
    def _pad(self, lst):
        arr = np.zeros((MAX_POSTS, POST_DIM), np.float32)
        msk = np.zeros(MAX_POSTS, np.float32)
        L = min(len(lst), MAX_POSTS)
        if L:
            arr[:L] = lst[:L]
            msk[:L] = 1
        return arr, msk

    def _rand_posts(self):
        arr = np.random.randn(MAX_POSTS, POST_DIM).astype(np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        return arr, np.ones(MAX_POSTS, np.float32)

    # --- Dataset std ---
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        f, t, lbl = self.data[idx]
        fp, fm = self._pad(self.posts.get(f, []))
        if t == "__RANDOM_USER__":
            tp, tm = self._rand_posts()
        else:
            tp, tm = self._pad(self.posts.get(t, []))

        # === 出現回数による weight ===
        if lbl == 1.0:
            freq = self.pos_f.get(t, self.avg_f)
            w = (self.avg_f / freq) ** self.alpha
        else:  # 負例側
            w = 1.0

        return (
            torch.tensor(fp),
            torch.tensor(fm),
            torch.tensor(tp),
            torch.tensor(tm),
            torch.tensor(lbl, dtype=torch.float32),
            torch.tensor(w, dtype=torch.float32),
        )


# --------------- Model ------------------------
class AttentionPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(POST_DIM, 8, batch_first=True)
        self.ln = nn.LayerNorm(POST_DIM)
        self.proj = nn.Sequential(
            nn.Linear(POST_DIM, ACCOUNT_DIM * 2),
            nn.GELU(),
            nn.Linear(ACCOUNT_DIM * 2, ACCOUNT_DIM),
        )
        self.sc = nn.Linear(POST_DIM, 1, bias=False)

    def forward(self, x, m):
        x = F.normalize(x, p=2, dim=-1)
        x = self.ln(x + self.mha(x, x, x, key_padding_mask=(m == 0))[0])
        w = F.softmax(
            self.sc(x).squeeze(-1).masked_fill(m == 0, -1e9), dim=-1
        )
        v = (x * w.unsqueeze(-1)).sum(1)
        return F.normalize(self.proj(v), p=2, dim=-1)


class FollowPred(nn.Module):
    def __init__(self):
        super().__init__()
        h = HIDDEN_DIM
        self.fe = nn.Sequential(nn.Linear(ACCOUNT_DIM, h), nn.GELU())
        self.te = nn.Sequential(nn.Linear(ACCOUNT_DIM, h), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(h * 3, h),
            nn.GELU(),
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Linear(h // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, f, t):
        f, t = self.fe(f), self.te(t)
        return self.head(torch.cat([f, t, f * t], -1)).squeeze(-1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = AttentionPool()
        self.fp = FollowPred()

    def forward(self, fp, fm, tp, tm):
        return self.fp(self.ap(fp, fm), self.ap(tp, tm))


# --------------- training loop ----------------
def train():
    dev = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    log(f"device={dev}")

    posts = load_posts()
    train_edges, val_edges = load_split_edges()
    log(f"train_edges={len(train_edges)}, val_edges={len(val_edges)}")

    # ---- followee positive 出現回数 ----
    pos_followee_counter = Counter(t for _, t in train_edges if t in posts)
    avg_freq = np.mean(list(pos_followee_counter.values()))
    log(f"avg positive followee freq = {avg_freq:.2f}")

    # ---- user pools ----
    train_users = {
        u for pair in train_edges for u in pair if u in posts
    }
    all_users = train_users | {u for pair in val_edges for u in pair if u in posts}
    train_users = list(train_users)
    all_users = list(all_users)

    # ---- validation固定 ----
    val_neg = simple_negative_sampling(all_users, int(len(val_edges) * NEG_RATIO))
    val_ds = SimplePairDS(
        val_edges, val_neg, posts, pos_followee_counter, avg_freq
    )
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ---- model / optim ----
    model = Model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best, wait = 1e9, 0
    for ep in range(1, NUM_EPOCHS + 1):
        # ---- epoch-wise negatives ----
        neg_pairs = simple_negative_sampling(
            train_users, int(len(train_edges) * NEG_RATIO)
        )
        neg_randoms = [
            (random.choice(train_users), "__RANDOM_USER__")
            for _ in range(int(len(train_edges) * RANDOM_NEG_RATIO))
        ]
        train_ds = SimplePairDS(
            train_edges,
            neg_pairs + neg_randoms,
            posts,
            pos_followee_counter,
            avg_freq,
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
        )

        # ---- train ----
        model.train()
        tot_loss = 0
        for fp, fm, tp, tm, lbl, w in tqdm(train_dl, desc=f"Epoch {ep}[TRAIN]"):
            fp, fm, tp, tm, lbl, w = [
                x.to(dev) for x in (fp, fm, tp, tm, lbl, w)
            ]
            pred = model(fp, fm, tp, tm)
            loss = (F.binary_cross_entropy(pred, lbl, reduction="none") * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(lbl)
        tr_loss = tot_loss / len(train_ds)

        # ---- val ----
        model.eval()
        vl_loss = 0
        preds, labels = [], []
        with torch.no_grad():
            for fp, fm, tp, tm, lbl, w in tqdm(val_dl, desc=f"Epoch {ep}[VAL]"):
                fp, fm, tp, tm, lbl, w = [
                    x.to(dev) for x in (fp, fm, tp, tm, lbl, w)
                ]
                p = model(fp, fm, tp, tm)
                vl_loss += (
                    F.binary_cross_entropy(p, lbl, reduction="none") * w
                ).mean().item() * len(lbl)
                preds.extend(p.cpu().numpy())
                labels.extend(lbl.cpu().numpy())
        vl_loss /= len(val_ds)
        acc = ((np.array(preds) >= 0.5) == np.array(labels)).mean()
        log(f"[{ep:03d}] train {tr_loss:.4f} | val {vl_loss:.4f} | acc {acc:.3f}")

        if vl_loss < best:
            best, wait = vl_loss, 0
            torch.save(model.state_dict(), MODEL_PATH)
            log("   saved best")
        else:
            wait += 1
            if wait >= EARLY_STOP:
                log("early stop")
                break


# --------------- main -------------------------
if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        log(f"Error: {e}")
        log(traceback.format_exc())