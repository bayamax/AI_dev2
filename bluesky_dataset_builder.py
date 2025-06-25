#!/usr/bin/env python3
"""
bluesky_dataset_builder.py  –  v0.2 (June 2025)
──────────────────────────────────────────────
Collect a **10 GB‑budget** Bluesky dataset:
  • account vectors  (OpenAI text‑embedding‑3‑large, 3072 dim fp32)
  • post vectors     (same embedding)
  • follow & like edges (uint32,uint32,uint8)
  • DID→aid  &  CID→pid lookup arrays for training‑time dereference

Changes (v0.2)
───────────────
* Removed deprecated `parse_car` import.  Now uses **py‑ipld‑car** for
  CAR decoding (CarReader).  `pip install py-ipld-car`.
* Added `requirements.txt` hint in docstring.
* Added sanity check on total vector count vs 10 GB limit.
"""

# ---------------------------------------------------------------------------
# Requirements
#   pip install atproto>=0.0.44  py-ipld-car  openai  numpy  tqdm  rich
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import openai
from atproto import Client  # Bluesky client (no auth needed for snapshot)
from ipld.car import CarReader  # from py-ipld-car
from tqdm import tqdm
from rich import print

# ------------------------- CONFIG CONSTANTS -------------------------------
EMBED_MODEL = "text-embedding-3-large"   # 3072‑dim fp32
BATCH_SIZE  = 96

VEC_SIZE    = 3072           # fp32 floats
BYTES_PER_VEC = VEC_SIZE * 4 # 12 288 B
GB_LIMIT    = 10             # hard cap for all npy output (rough)

# --------------------------- UTILITIES ------------------------------------

def estimate_total_gb(acc_ct: int, post_ct: int) -> float:
    return (acc_ct + post_ct) * BYTES_PER_VEC / (1024 ** 3)


def embed_texts(texts):
    resp = openai.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


# ---------------------------- MAIN CLASS ----------------------------------

class BlueskyCollector:
    def __init__(self, out_dir: Path, max_accounts: int, max_posts: int,
                 max_followees: int, max_likes: int):
        self.out_dir = out_dir
        self.max_accounts = max_accounts
        self.max_posts = max_posts
        self.max_followees = max_followees
        self.max_likes = max_likes

        self.client = Client()  # unauth snapshot api
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # memory‑mapped npy writers (growable via resize + append)
        self.acc_vecs = None  # type: np.ndarray
        self.post_vecs = None  # type: np.ndarray
        self.edges     = []   # list[Tuple[int,int,int]]  — keep in RAM, flush at end

        self.did2aid: Dict[str, int] = {}
        self.cid2pid: Dict[str, int] = {}

        self.acc_counter = 0
        self.post_counter = 0

    # ------------------------------------------------------------------
    # Low‑level helpers
    # ------------------------------------------------------------------
    def _assign_account(self, did: str) -> int:
        if did in self.did2aid:
            return self.did2aid[did]
        aid = self.acc_counter
        self.did2aid[did] = aid
        self.acc_counter += 1
        return aid

    def _assign_post(self, cid: str) -> int:
        if cid in self.cid2pid:
            return self.cid2pid[cid]
        pid = self.post_counter
        self.cid2pid[cid] = pid
        self.post_counter += 1
        return pid

    # ------------------------------------------------------------------
    def _ensure_vec_arrays(self):
        """Create or grow mem‑mapped numpy arrays."""
        acc_path  = self.out_dir / "accounts.npy"
        post_path = self.out_dir / "posts.npy"

        if self.acc_vecs is None:
            self.acc_vecs = np.lib.format.open_memmap(
                acc_path, dtype=np.float32, mode="w+",
                shape=(self.max_accounts, VEC_SIZE))
        if self.post_vecs is None:
            self.post_vecs = np.lib.format.open_memmap(
                post_path, dtype=np.float32, mode="w+",
                shape=(self.max_accounts * self.max_posts, VEC_SIZE))

    # ------------------------------------------------------------------
    def collect(self, seed_did: str):
        self._ensure_vec_arrays()

        q: deque[str] = deque([seed_did])
        visited: Set[str] = set()

        pbar = tqdm(total=self.max_accounts, desc="Accounts", unit="acc")

        while q and self.acc_counter < self.max_accounts:
            did = q.popleft()
            if did in visited:
                continue
            visited.add(did)

            # Fetch repo snapshot (CAR bytes)
            try:
                car_bytes = self.client.com.atproto.sync.get_repo({"did": did})
            except Exception as e:
                print(f"[yellow]Warning[/] get_repo failed for {did}: {e}")
                continue

            aid = self._assign_account(did)

            # Decode CAR and collect records
            reader = CarReader.from_bytes(car_bytes)
            posts_text: list[str] = []
            followees: list[str] = []
            likes: list[str] = []  # post CIDs

            for block in reader.iter_blocks():
                data = block.json
                if not data:  # skip non‑json (e.g., images)
                    continue

                rtype = data.get("type")
                if rtype == "app.bsky.feed.post":
                    if len(posts_text) < self.max_posts:
                        txt = data.get("text", "")
                        posts_text.append(txt)
                elif rtype == "app.bsky.graph.follow":
                    if len(followees) < self.max_followees:
                        followees.append(data.get("subject"))
                elif rtype == "app.bsky.feed.like":
                    if len(likes) < self.max_likes:
                        likes.append(data["subject"]["cid"])

            # --- Account Vector (mean of self posts) ----------------
            if posts_text:
                acc_emb = embed_texts(posts_text).mean(axis=0)
            else:
                acc_emb = np.zeros(VEC_SIZE, dtype=np.float32)
            self.acc_vecs[aid] = acc_emb

            # --- Post vectors --------------------------------------
            if posts_text:
                p_vecs = embed_texts(posts_text)
                for v, txt in zip(p_vecs, posts_text):
                    pid = self._assign_post(hash(txt))  # crude cid replacement
                    self.post_vecs[pid] = v
                    self.edges.append((aid, 1, pid))  # like‑style edge self→post

            # --- Follow edges & queue ------------------------------
            for fdid in followees:
                dst = self._assign_account(fdid)
                self.edges.append((aid, 0, dst))
                if dst >= self.max_accounts:
                    continue
                if fdid not in visited:
                    q.append(fdid)

            pbar.update(1)

            # --- Check size budget --------------------------------
            if estimate_total_gb(self.acc_counter, self.post_counter) > GB_LIMIT:
                print("[red]GB limit reached. Stopping collection.[/]")
                break

        pbar.close()
        self._flush_edges()
        self._flush_mappings()
        print("[green]Collection complete.[/]")

    # ------------------------------------------------------------------
    def _flush_edges(self):
        arr = np.array(self.edges, dtype=[("src", "<u4"), ("typ", "u1"), ("dst", "<u4")])
        np.savez_compressed(self.out_dir / "edges.npz", edges=arr)

    def _flush_mappings(self):
        (self.out_dir / "did2aid.json").write_text(json.dumps(self.did2aid))
        np.save(self.out_dir / "cid2pid.npy", np.array(list(self.cid2pid.items()), dtype="<U64,<u4"))

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bluesky 10 GB dataset crawler")
    parser.add_argument("--seed-did", default="did:plc:jl72kzcyrcdjdxtahcnj5n6i", help="seed account DID (default: official bluesky team)")
    parser.add_argument("--output-dir", default="./data10gb", help="root directory to write .npy/.npz files")
    parser.add_argument("--max-accounts", type=int, default=60000)
    parser.add_argument("--max-posts", type=int, default=10)
    parser.add_argument("--max-followees", type=int, default=20)
    parser.add_argument("--max-likes", type=int, default=15)
    parser.add_argument("--openai-model", default="text-embedding-3-large")
    args = parser.parse_args()

    print("[i] effective parameters:")
    for k,v in vars(args).items():
        print(f"   {k}: {v}")

    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise SystemExit("[!] set OPENAI_API_KEY env var first")

    crawl(args)