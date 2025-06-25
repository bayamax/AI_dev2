#!/usr/bin/env python3
"""
bluesky_dataset_builder.py
──────────────────────────
Collect a *10 GB‑budget* Bluesky dataset containing:
  • account vectors (OpenAI text‑embedding‑3‑large, 3072 dim fp32)
  • post vectors (same embedding, 3072 dim fp32)
  • follow & like edges (uint32 src, uint32 dst, uint8 type)
  • DID→aid and CID→pid lookup arrays for training‑time dereference

The script performs a bounded breadth‑first search:
  • Seed DID list (default: creator of handle supplied via --seed-handle)
  • Traverse up to --max-accounts (default 60 000)
  • At each account: keep latest --max-posts (10) & latest --max-followees (20)
  • Record likes (max 15) for edge signal but **do NOT embed liked posts**

Total vector cap: 60 k Acc + 600 k Post ≈ 9 GB
Disk layout (all npy/npz):
  vectors/accounts.npy   float32 (N_acc, 3072)
  vectors/posts.npy      float32 (N_post, 3072)
  graphs/edges.npz       arrays src,dst,type
  maps/did2aid.json      {did:str → aid:int}
  maps/cid2pid.npy       uint32 (post_count,)  # index = pid, value = len(CID SHA256)==0 ⇒ tombstone

Dependencies:
  pip install atproto openai numpy tqdm requests
  export OPENAI_API_KEY=sk-...

Run:
  python bluesky_dataset_builder.py --seed-handle your.bsky.social \
         --max-accounts 60000 --out-dir /datasets/bluesky10g
"""

import argparse, os, json, sys, functools, time, hashlib, asyncio, base64
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Set

import numpy as np
from tqdm import tqdm
from atproto import Client, parse_car
import openai

EMBED_BATCH = 96  # OpenAI max for large embedding
EMBED_MODEL = "text-embedding-3-large"
VEC_DIM = 3072

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def sha256_b64(s: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(s.encode()).digest()).decode()


def eager_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Bluesky fetch helpers (sync for simplicity; rate‑limit via semaphore)
# ---------------------------------------------------------------------------

class BlueskyFetcher:
    def __init__(self, concurrency: int = 6):
        self.client = Client()
        self.sem = asyncio.Semaphore(concurrency)

    async def fetch_repo(self, did: str) -> bytes:
        async with self.sem:
            return self.client.com.atproto.sync.get_repo({"did": did})

    def list_repos(self, limit: int = 1):
        return self.client.com.atproto.sync.list_repos({"limit": limit})["repos"]


# ---------------------------------------------------------------------------
# Embedding – batched OpenAI calls with simple exponential back‑off
# ---------------------------------------------------------------------------

async def embed_texts(texts: List[str]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for i in range(0, len(texts), EMBED_BATCH):
        chunk = texts[i : i + EMBED_BATCH]
        for attempt in range(5):
            try:
                resp = openai.embeddings.create(model=EMBED_MODEL, input=chunk)
                out.extend([np.array(e.embedding, dtype=np.float32) for e in resp.data])
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"OpenAI error {e}, retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
        else:
            raise RuntimeError("OpenAI failed after retries")
    return out


# ---------------------------------------------------------------------------
# Dataset builder class
# ---------------------------------------------------------------------------

class DatasetBuilder:
    def __init__(
        self,
        out_dir: str,
        max_accounts: int = 60000,
        max_posts_per_acc: int = 10,
        max_followees: int = 20,
    ):
        self.out_dir = out_dir
        eager_mkdir(out_dir)
        eager_mkdir(os.path.join(out_dir, "vectors"))
        eager_mkdir(os.path.join(out_dir, "graphs"))
        eager_mkdir(os.path.join(out_dir, "maps"))

        self.max_accounts = max_accounts
        self.max_posts = max_posts_per_acc
        self.max_followees = max_followees

        # Dynamic containers
        self.did2aid: Dict[str, int] = {}
        self.cid2pid: Dict[str, int] = {}
        self.account_vecs: List[np.ndarray] = []
        self.post_vecs: List[np.ndarray] = []
        self.edges_src: List[int] = []
        self.edges_dst: List[int] = []
        self.edges_type: List[int] = []  # 0 follow, 1 like

    # --------------------------------------------------
    def _assign_aid(self, did: str) -> int:
        if did not in self.did2aid:
            self.did2aid[did] = len(self.did2aid)
        return self.did2aid[did]

    def _assign_pid(self, cid: str) -> int:
        if cid not in self.cid2pid:
            self.cid2pid[cid] = len(self.cid2pid)
        return self.cid2pid[cid]

    # --------------------------------------------------
    async def process_account(self, did: str, fetcher: BlueskyFetcher):
        # fetch repo (CAR bytes)
        car_bytes = await fetcher.fetch_repo(did)
        records = list(parse_car(car_bytes))

        # posts for embedding
        posts_text: List[str] = []
        posts_cids: List[str] = []
        follows: List[str] = []
        likes: List[Tuple[str, str]] = []  # (post_cid, author_did)

        for rec in records:
            if rec["type"] == "app.bsky.feed.post":
                if len(posts_text) < self.max_posts:
                    text = rec.value.get("text", "")
                    posts_text.append(text)
                    posts_cids.append(rec.cid)
            elif rec["type"] == "app.bsky.graph.follow":
                if len(follows) < self.max_followees:
                    follows.append(rec.value["subject"])
            elif rec["type"] == "app.bsky.feed.like":
                if len(likes) < 15:
                    likes.append((rec.value["subject"], rec.value.get("author", "")))

        # embed posts → PostVecs and AccVec (mean)
        post_vecs = await embed_texts(posts_text) if posts_text else []
        if post_vecs:
            acc_vec = np.mean(post_vecs, axis=0)
        else:
            # empty account; use zeros
            acc_vec = np.zeros(VEC_DIM, dtype=np.float32)

        aid = self._assign_aid(did)
        self.account_vecs.append(acc_vec)

        for cid, vec in zip(posts_cids, post_vecs):
            pid = self._assign_pid(cid)
            self.post_vecs.append(vec)
            # implicit edge author→post not stored; can be derived

        # store follow edges
        for target_did in follows:
            dst = self._assign_aid(target_did)
            self.edges_src.append(aid)
            self.edges_dst.append(dst)
            self.edges_type.append(0)

        # store like edges (author→post_author)
        for post_cid, post_author in likes:
            if post_author and post_author in self.did2aid:
                dst = self.did2aid[post_author]
                self.edges_src.append(aid)
                self.edges_dst.append(dst)
                self.edges_type.append(1)

    # --------------------------------------------------
    async def crawl(self, seed_dids: List[str]):
        fetcher = BlueskyFetcher()
        queue: deque[str] = deque(seed_dids)
        visited: Set[str] = set()

        pbar = tqdm(total=self.max_accounts, desc="Accounts processed")
        while queue and len(self.did2aid) < self.max_accounts:
            did = queue.popleft()
            if did in visited:
                continue
            visited.add(did)
            try:
                await self.process_account(did, fetcher)
                pbar.update(1)
            except Exception as e:
                print(f"Error processing {did}: {e}", file=sys.stderr)
                continue
            # enqueue newly discovered follows capped by max_followees
            new_follows = [d for d in (await fetcher.client.com.atproto.sync.get_repo({"did": did})) if d not in visited]
            for f in new_follows[: self.max_followees]:
                if f not in visited:
                    queue.append(f)
        pbar.close()

    # --------------------------------------------------
    def save(self):
        print("Saving numpy arrays …")
        np.save(os.path.join(self.out_dir, "vectors", "accounts.npy"), np.stack(self.account_vecs, dtype=np.float32))
        np.save(os.path.join(self.out_dir, "vectors", "posts.npy"), np.stack(self.post_vecs, dtype=np.float32))
        np.savez_compressed(
            os.path.join(self.out_dir, "graphs", "edges.npz"),
            src=np.array(self.edges_src, dtype=np.uint32),
            dst=np.array(self.edges_dst, dtype=np.uint32),
            etype=np.array(self.edges_type, dtype=np.uint8),
        )
        with open(os.path.join(self.out_dir, "maps", "did2aid.json"), "w") as f:
            json.dump(self.did2aid, f)
        np.save(os.path.join(self.out_dir, "maps", "cid2pid.npy"), np.array(self.cid2pid))
        print("All done ✔")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli():
    parser = argparse.ArgumentParser(description="Collect Bluesky dataset (10 GB budget)")
    parser.add_argument("--seed-handle", required=True, help="Initial account handle, e.g. foo.bsky.social")
    parser.add_argument("--out-dir", default="./dataset10g", help="Output directory")
    parser.add_argument("--max-accounts", type=int, default=60000)
    parser.add_argument("--max-posts", type=int, default=10)
    parser.add_argument("--max-followees", type=int, default=20)
    args = parser.parse_args()

    # resolve handle → DID
    client = Client()
    did = client.com.atproto.identity.resolve_handle({"handle": args.seed_handle})["did"]

    builder = DatasetBuilder(
        out_dir=args.out_dir,
        max_accounts=args.max_accounts,
        max_posts_per_acc=args.max_posts,
        max_followees=args.max_followees,
    )
    asyncio.run(builder.crawl([did]))
    builder.save()


if __name__ == "__main__":
    cli()
