#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_followers_from_sequence.py
────────────────────────────────────────────────────────
■ 使い方（どれか 1 つ）
  1) 対話入力        : python predict_followers_from_sequence.py
        → 1 行 = 1 投稿を入力、空行 2 回で確定
  2) --text          : python ... --text "投稿1" "投稿2" --topk 15
  3) --file          : python ... --file posts.txt        (改行区切り)
────────────────────────────────────────────────────────
フロー
 1. 入力投稿すべてを OpenAI Embedding (3072d) へ
 2. (N≤50, 3072) を AttentionPool → 128 次元「仮想アカウント」ベクトル
 3. 学習済みユーザ全員と FollowPred でスコア計算
 4. 『この投稿群をするアカウントをフォローしそうな既存アカウント』Top-K を表示
"""

# ====== ★ OpenAI API キーをここに入力してください ======================
OPENAI_API_KEY = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# =======================================================================

from pathlib import Path
import argparse, csv, os, sys, time, textwrap

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import openai                            # pip install openai>=1.0.0

# ---------- PATH & 定数 (学習時と合わせる) ------------------------------
BASE_DIR   = Path('/workspace/workspace/vast')   # 必要なら変更
POSTS_NPY  = BASE_DIR / 'aggregated_posting_vectors.npy'
EDGES_CSV  = BASE_DIR / 'edges.csv'
CKPT_PATH  = BASE_DIR / 'attention_rank_out' / 'rank_follow_model.pt'

POST_DIM, ACCOUNT_DIM, HIDDEN_DIM = 3072, 128, 256
MAX_POSTS                        = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------ モデル定義（学習コードと完全一致） -------------------
class AttentionPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha  = nn.MultiheadAttention(POST_DIM, 8, batch_first=True)
        self.ln   = nn.LayerNorm(POST_DIM)
        self.proj = nn.Sequential(
            nn.Linear(POST_DIM, ACCOUNT_DIM * 2), nn.GELU(),
            nn.Linear(ACCOUNT_DIM * 2, ACCOUNT_DIM))
        self.sc   = nn.Linear(POST_DIM, 1, bias=False)

    def forward(self, x, m):           # x:(B,N,3072)  m:(B,N)
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

# ------------------ ユーティリティ --------------------------------------
def pad_posts(vecs: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    vecs : List[(3072,)]
    戻り値: fp:(MAX_POSTS,3072)  fm:(MAX_POSTS,)
    """
    fp = np.zeros((MAX_POSTS, POST_DIM), np.float32)
    fm = np.zeros(MAX_POSTS, np.float32)
    L = min(len(vecs), MAX_POSTS)
    if L:
        fp[:L] = np.stack(vecs[:L])
        fm[:L] = 1
    return torch.tensor(fp), torch.tensor(fm)

def build_post_cache(posts):
    cache = {}
    for u, vecs in posts.items():
        fp, fm = pad_posts(vecs)       # 学習時の投稿リスト
        cache[u] = (fp, fm)
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
    print("⏎ を 2 回で入力終了（Ctrl+D でも可）")
    lines, blanks = [], 0
    while True:
        try:
            line = input(prompt)
        except EOFError:
            break
        if line.strip() == "":
            blanks += 1
            if blanks == 2:
                break
        else:
            blanks = 0
            lines.append(line)
    return lines

# ------------------ メイン処理 -----------------------------------------
def main(texts, topk):
    # ① モデル & 既存ユーザベクトル
    posts = np.load(POSTS_NPY, allow_pickle=True).item()
    cache = build_post_cache(posts)
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()
    users, V = compute_all_vectors(model, cache)

    # ② 入力投稿群 → Embedding → AttentionPool
    emb_list = [get_embedding(t) for t in texts]
    fp, fm = pad_posts(emb_list)
    with torch.no_grad():
        t_vec = model.ap(fp.unsqueeze(0).to(DEVICE).float(),
                         fm.unsqueeze(0).to(DEVICE).float())  # (1,128)

    # ③ フォロワー候補スコア
    with torch.no_grad():
        scores = model.fp(V, t_vec.repeat(len(V), 1)).cpu().numpy()

    cand = [(users[i], float(scores[i])) for i in range(len(users))]
    cand.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "="*80)
    print("INPUT POSTS ({}件):".format(len(texts)))
    print("‾"*80)
    for t in texts:  # 簡潔に折り返し表示
        print(textwrap.fill(t, 70))
    print("-"*80)
    for r, (u, sc) in enumerate(cand[:topk], 1):
        print(f"{r:2d}. {u:25s} | score {sc:+.4f}")

# ------------------ CLI ------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--text", nargs='+', help="スペース区切りで複数投稿")
    g.add_argument("--file", help="改行区切りで複数投稿")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    if OPENAI_API_KEY.startswith("sk-XXXX"):
        sys.exit("★ OPENAI_API_KEY を設定してください ★")
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