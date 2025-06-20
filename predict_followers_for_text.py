#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_followers_for_text.py
────────────────────────────────────────────────────────
任意のテキスト（複数可・対話入力可）から：
  1. OpenAI 埋め込み (3072d)
  2. AttentionPool → 128d
  3. 既存ユーザとの FollowPred で「フォローしそうなアカウント」を Top-K
────────────────────────────────────────────────────────
"""

# ====== ★ OpenAI API キーをここに入力してください ======================
OPENAI_API_KEY = ""
# =======================================================================

from pathlib import Path
import argparse, csv, os, sys, time, textwrap

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import openai                              # pip install openai>=1.0.0

# ---------- PATH & CONST ------------------------------------------------
BASE_DIR   = Path('/workspace/workspace/vast')        # 適宜変更
POSTS_NPY  = BASE_DIR / 'aggregated_posting_vectors.npy'
EDGES_CSV  = BASE_DIR / 'edges.csv'
CKPT_PATH  = BASE_DIR / 'attention_rank_out' / 'rank_follow_model.pt'

POST_DIM, ACCOUNT_DIM, HIDDEN_DIM = 3072, 128, 256
MAX_POSTS                        = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- Model (学習コードと同じ) ------------------------------------
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

# ---------- Utility -----------------------------------------------------
def pad_posts(arr):
    out = np.zeros((MAX_POSTS, POST_DIM), np.float32)
    msk = np.zeros(MAX_POSTS, np.float32)
    out[0] = arr;  msk[0] = 1
    return torch.tensor(out), torch.tensor(msk)

def build_post_cache(posts):
    cache = {}
    for u, vecs in posts.items():
        fp = np.zeros((MAX_POSTS, POST_DIM), np.float32)
        fm = np.zeros(MAX_POSTS, np.float32)
        L = min(len(vecs), MAX_POSTS)
        if L:
            fp[:L] = vecs[:L];  fm[:L] = 1
        cache[u] = (torch.tensor(fp), torch.tensor(fm))
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

def get_embedding(text: str) -> np.ndarray:
    resp = openai.embeddings.create(
        model="text-embedding-3-large",
        input=text.strip())
    return np.array(resp.data[0].embedding, dtype=np.float32)

def read_stdin_multi(prompt="> ") -> list[str]:
    print("⏎ 2 回で確定してください（Ctrl+D でも終了）")
    lines, blank = [], 0
    while True:
        try:
            line = input(prompt)
        except EOFError:
            break
        if line.strip() == "":
            blank += 1
            if blank == 2:
                break
        else:
            blank = 0
            lines.append(line)
    return lines

# ---------- Main --------------------------------------------------------
def main(texts, topk):
    posts = np.load(POSTS_NPY, allow_pickle=True).item()
    cache = build_post_cache(posts)

    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()

    users, V = compute_all_vectors(model, cache)
    edge_set = set(load_edges())

    for text in texts:
        emb = get_embedding(text)
        fp, fm = pad_posts(emb)

        with torch.no_grad():
            t_vec = model.ap(fp.unsqueeze(0).to(DEVICE).float(),
                             fm.unsqueeze(0).to(DEVICE).float())
            scores = model.fp(V, t_vec.repeat(len(V), 1)).cpu().numpy()

        cand = [(u, float(scores[i])) for i, u in enumerate(users)
                if (u, "__dummy__") not in edge_set]   # 既フォロー情報無し
        cand.sort(key=lambda x: x[1], reverse=True)
        print("\n" + "="*80)
        print(textwrap.fill(text, 70))
        print("-"*80)
        for rank, (u, sc) in enumerate(cand[:topk], 1):
            print(f"{rank:2d}. {u:25s} | score {sc:+.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--text", nargs='+', help="スペース区切りで複数テキスト")
    g.add_argument("--file", help="改行区切りで複数テキスト")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    if OPENAI_API_KEY.startswith("sk-XXXX"):
        sys.exit("★OPENAI_API_KEY を設定してください★")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    openai.api_key = OPENAI_API_KEY

    if args.text:
        texts = args.text
    elif args.file:
        texts = [l.rstrip("\n") for l in open(args.file)]
    else:
        texts = read_stdin_multi()

    t0 = time.time()
    main(texts, args.topk)
    print(f"\nDone in {time.time()-t0:.1f}s  (device={DEVICE})")