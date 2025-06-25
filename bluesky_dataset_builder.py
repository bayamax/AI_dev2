#!/usr/bin/env python3
"""bluesky_dataset_builder.py – v0.5 (June 2025)

Collect up to 60 k accounts, each up to 10 authored posts and 20 followees,
plus up to 15 liked posts, staying within ~10 GB disk for 3072‑dim fp32 vectors.

Outputs
-------
data_dir/
    accounts.npy      (N_accounts, 3072) float32
    posts.npy         (N_posts,    3072) float32
    edges.npz         compressed npz with {'src':uint32,'dst':uint32,'type':uint8}
    did2aid.json      DID  -> account row id
    cid2pid.npy       CID  -> post    row id  (uint32)

Dependencies
------------
pip install 'atproto>=0.0.61' openai numpy tqdm

Environment
-----------
export OPENAI_API_KEY=sk-...

"""

import argparse, json, os, struct, sys, time
from collections import deque, defaultdict

import numpy as np
from tqdm import tqdm
from atproto import Client
import openai

# ---------- hyper‑params (default) ----------
MAX_ACCOUNTS   = 60_000
MAX_POSTS      = 10
MAX_FOLLOWEES  = 20
MAX_LIKES      = 15
BATCH_EMBED    = 96
EMBED_DIM      = 3072
# -------------------------------------------

def embed_texts(texts):
    # OpenAI batch embedding call
    resp = openai.Embedding.create(
            model="text-embedding-3-large",
            input=texts)
    return [np.array(d['embedding'], dtype=np.float32) for d in resp['data']]

def ensure_arrays(path, rows, dim):
    if os.path.exists(path):
        arr = np.load(path, mmap_mode='r+')
        return arr
    arr = np.memmap(path, dtype=np.float32, mode='w+', shape=(rows, dim))
    arr.flush()
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed-did', default='did:plc:jl72kzcyrcdjdxtahcnj5n6i')
    ap.add_argument('--output-dir', default='./data10gb')
    ap.add_argument('--max-accounts', type=int, default=MAX_ACCOUNTS)
    ap.add_argument('--max-posts', type=int, default=MAX_POSTS)
    ap.add_argument('--max-followees', type=int, default=MAX_FOLLOWEES)
    args = ap.parse_args()

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # Pre‑allocate with generous upper bounds; we'll truncate later
    acc_arr  = ensure_arrays(f"{out}/accounts.npy", args.max_accounts, EMBED_DIM)
    post_arr = ensure_arrays(f"{out}/posts.npy",   args.max_accounts*args.max_posts, EMBED_DIM)

    edges_src = []
    edges_dst = []
    edges_typ = []

    did2aid = {}
    cid2pid = {}

    client = Client()
    client.sessions.create(args.seed_did)  # anonymous session

    q = deque([args.seed_did])
    bar = tqdm(total=args.max_accounts, desc="accounts")

    next_post_row = 0
    aid_counter   = 0

    while q and aid_counter < args.max_accounts:
        did = q.popleft()
        if did in did2aid:
            continue

        # allocate account id
        aid = aid_counter
        did2aid[did] = aid
        aid_counter += 1
        bar.update(1)

        # ---------- fetch and embed posts ----------
        recs = client.com.atproto.repo.list_records(
            {'repo': did, 'collection': 'app.bsky.feed.post', 'limit': args.max_posts})['records']

        texts = [r['value'].get('text', '')[:2048] for r in recs]
        if texts:
            vecs = embed_texts(texts)
            for vec, rec in zip(vecs, recs):
                cid = rec['cid']
                if cid in cid2pid:
                    pid = cid2pid[cid]
                else:
                    pid = next_post_row
                    post_arr[pid] = vec
                    cid2pid[cid] = pid
                    next_post_row += 1
                edges_src.append(aid); edges_dst.append(pid); edges_typ.append(1)  # like‑type edge from self‑post

        # ---------- followees ----------
        follows = client.app.bsky.graph.get_follows({'actor': did, 'limit': args.max_followees})['follows']
        for f in follows:
            fdid = f['did']
            edges_src.append(aid); edges_dst.append(did2aid.get(fdid, 0)); edges_typ.append(0)
            if fdid not in did2aid and len(did2aid) < args.max_accounts:
                q.append(fdid)

    bar.close()
    # ---------- save edges ----------
    edges = {'src': np.array(edges_src, dtype=np.uint32),
             'dst': np.array(edges_dst, dtype=np.uint32),
             'type': np.array(edges_typ, dtype=np.uint8)}
    np.savez_compressed(f"{out}/edges.npz", **edges)

    # ---------- truncate arrays ----------
    np.save(f"{out}/accounts.npy", np.memmap(f"{out}/accounts.npy", mode='r', dtype=np.float32, shape=(aid_counter, EMBED_DIM)))
    np.save(f"{out}/posts.npy",    np.memmap(f"{out}/posts.npy",    mode='r', dtype=np.float32, shape=(next_post_row, EMBED_DIM)))

    with open(f"{out}/did2aid.json", 'w') as fp:
        json.dump(did2aid, fp)

    np.save(f"{out}/cid2pid.npy", np.array([(cid, pid) for cid, pid in cid2pid.items()], dtype=[('cid','U64'),('pid','u4')]))

    print(f"Done. Accounts {aid_counter}, Posts {next_post_row}, Edges {len(edges_src)}")


if __name__ == "__main__":
    main()
