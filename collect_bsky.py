#!/usr/bin/env python
"""
collect_bsky.py  –  Bluesky 収集スクリプト（ワンファイル版）

Python ≥3.10  /  vast.ai 等 Linux 環境で動作確認
依存:  pip install aiohttp openai atproto numpy torch tqdm

────────────────────────────────────────────────────────────
・各アカウントから
    投稿 10–40 件
    いいね対象投稿 10–40 件
    フォロー 20 件以上
  を取得して OpenAI text-embedding-3-large (1536 fp32) へ即埋め込み
・ベクトルは mmap .npy、メタは SQLite に保存
・途中停止 → 再実行で自動リジューム
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import sqlite3
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import aiohttp
import numpy as np
import openai
from aiohttp import ClientResponseError
from tqdm.asyncio import tqdm_asyncio

# ========= ユーザ設定 =========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or sys.exit("OPENAI_API_KEY 未設定")
SEED_HANDLES = [
    # 実在ユーザを複数指定すると探索が途切れにくい
    "ngntrtr.bsky.social",
    "uishig.bsky.social",
    "mikapikazompz.bsky.social",
    "purinharumaki.bsky.social",
    "sora-sakurai.bsky.social",
]

BASE_ENDPOINT = "https://public.api.bsky.app/xrpc"

EMB_MODEL = "text-embedding-3-large"
EMB_DIM = 1536
BATCH_EMB = 96

MIN_POSTS = 10
MIN_LIKES = 10
MIN_FOLLOWS = 20
MAX_POSTS_ACC = 40
MAX_ACCOUNTS = 55_000

OUT_DIR = Path("dataset")
OUT_DIR.mkdir(exist_ok=True)

LOGLEVEL = logging.INFO
# =============================


# ========= I/O =========
class EmbeddingStore:
    """1536-dim fp32 ベクトルを連続保存 / メタは SQLite"""

    def __init__(self, n_rows: int = 2_500_000):
        self.db = sqlite3.connect(OUT_DIR / "meta.db")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS post(
              idx      INTEGER PRIMARY KEY,
              account  TEXT,
              post_uri TEXT,
              kind     TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS post_uq ON post(account, post_uri, kind);
            """
        )

        path = OUT_DIR / "embeddings.npy"
        if not path.exists():
            np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32, shape=(n_rows, EMB_DIM)
            )[:] = 0.0
        self.arr = np.lib.format.open_memmap(path, mode="r+", dtype=np.float32)
        self.next = (
            self.db.execute("SELECT MAX(idx) FROM post").fetchone()[0] or -1
        ) + 1

    def add(self, metas: List[Tuple[str, str, str]], vecs: np.ndarray) -> None:
        n = len(metas)
        if self.next + n > self.arr.shape[0]:
            raise RuntimeError("mmap full – サイズを拡張してください")

        self.arr[self.next : self.next + n] = vecs
        self.db.executemany(
            "INSERT OR IGNORE INTO post(idx,account,post_uri,kind) VALUES (?,?,?,?)",
            [(self.next + i, *m) for i, m in enumerate(metas)],
        )
        self.db.commit()
        self.next += n


class GraphStore:
    """フォローグラフを SQLite に格納"""

    def __init__(self):
        self.db = sqlite3.connect(OUT_DIR / "graph.db")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS follow(
              src TEXT,
              dst TEXT,
              UNIQUE(src,dst)
            );
            CREATE INDEX IF NOT EXISTS idx_src ON follow(src);
            """
        )

    def add(self, src: str, dsts: Iterable[str]) -> None:
        self.db.executemany(
            "INSERT OR IGNORE INTO follow(src,dst) VALUES (?,?)",
            [(src, d) for d in dsts],
        )
        self.db.commit()


# ========= Bluesky API =========
class BlueskyAPI:
    """必要最小限の GET を aiohttp で叩く"""

    def __init__(self, session: aiohttp.ClientSession):
        self.sess = session

    async def _get(self, endpoint: str, **params) -> Dict[str, Any]:
        url = f"{BASE_ENDPOINT}/{endpoint}"
        while True:
            r = await self.sess.get(url, params=params, timeout=30)
            if r.status in (502, 503, 504):
                await asyncio.sleep(5)
                continue
            if r.status == 429:
                await asyncio.sleep(15)
                continue
            r.raise_for_status()
            return await r.json()

    async def posts(self, actor: str, limit: int = MAX_POSTS_ACC):
        js = await self._get(
            "app.bsky.feed.getAuthorFeed", actor=actor, limit=limit
        )
        for item in js.get("feed", []):
            record = item["post"]["record"]
            yield item["post"]["uri"], record.get("text", "")

    async def follows(self, actor: str, limit: int = 1000) -> List[str]:
        js = await self._get(
            "app.bsky.graph.getFollows", actor=actor, limit=limit
        )
        return [u["handle"] for u in js.get("follows", [])]

    async def like_uris(
        self, actor: str, limit: int = MAX_POSTS_ACC
    ) -> List[str]:
        try:
            js = await self._get(
                "com.atproto.repo.listRecords",
                repo=actor,
                collection="app.bsky.feed.like",
                limit=limit,
            )
        except ClientResponseError as e:
            if e.status == 404:  # Like レコードなし
                return []
            raise
        return [rec["value"]["subject"]["uri"] for rec in js.get("records", [])]

    async def posts_by_uri(self, uris: List[str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for chunk in [uris[i : i + 25] for i in range(0, len(uris), 25)]:
            js = await self._get("app.bsky.feed.getPosts", uris=chunk)
            for p in js.get("posts", []):
                out[p["uri"]] = p["record"].get("text", "")
        return out


# ========= Embedding =========
openai.api_key = OPENAI_API_KEY


async def embed_texts(texts: List[str]) -> np.ndarray:
    for _ in range(3):
        try:
            res = await openai.embeddings.async_create(
                model=EMB_MODEL, input=texts
            )
            return np.asarray(
                [d["embedding"] for d in res.data], dtype=np.float32
            )
        except openai.RateLimitError:
            await asyncio.sleep(5)
    raise RuntimeError("OpenAI embed failed (3 retries)")


async def embed_stream(stream, store: EmbeddingStore) -> None:
    """stream: async iterator -> (meta, text)"""
    buf_meta: List[Tuple[str, str, str]] = []
    buf_txt: List[str] = []

    async for meta, txt in stream:
        if not txt:
            continue
        buf_meta.append(meta)
        buf_txt.append(txt[:2048])  # 文字数上限
        if len(buf_txt) >= BATCH_EMB:
            store.add(buf_meta, await embed_texts(buf_txt))
            buf_meta, buf_txt = [], []

    if buf_txt:
        store.add(buf_meta, await embed_texts(buf_txt))


# ========= Crawler =========
class Crawler:
    def __init__(
        self, api: BlueskyAPI, estore: EmbeddingStore, gstore: GraphStore
    ):
        self.api, self.E, self.G = api, estore, gstore
        self.seen_accounts: set[str] = set()
        self.q: deque[str] = deque()

    async def seed(self, handles: List[str]) -> None:
        self.q.extend(handles)

    async def run(self) -> None:
        pbar = tqdm_asyncio(total=MAX_ACCOUNTS, desc="accounts", unit="")
        while self.q and len(self.seen_accounts) < MAX_ACCOUNTS:
            actor = self.q.popleft()
            if actor in self.seen_accounts:
                continue
            try:
                progressed = await self.process_actor(actor)
                if progressed:
                    self.seen_accounts.add(actor)
                    pbar.update(1)
            except Exception as e:
                logging.warning(f"{actor}: {e}")

    async def process_actor(self, actor: str) -> bool:
        # 投稿
        posts = [p async for p in self.api.posts(actor)]
        # フォロー
        follows = await self.api.follows(actor)
        # いいね
        like_uris = await self.api.like_uris(actor)
        like_posts = await self.api.posts_by_uri(like_uris[:MAX_POSTS_ACC])

        # グラフは常に保存し、隣接も enqueue
        self.G.add(actor, follows)
        random.shuffle(follows)
        self.q.extend(follows[:50])

        # 閾値を満たさない場合は探索だけ継続
        if (
            len(posts) < MIN_POSTS
            or len(like_posts) < MIN_LIKES
            or len(follows) < MIN_FOLLOWS
        ):
            return False  # count しない

        # === 埋め込み保存 ===
        async def gen():
            for uri, text in posts[:MAX_POSTS_ACC]:
                yield (actor, uri, "post"), text
            for uri, text in like_posts.items():
                yield (actor, uri, "like"), text

        await embed_stream(gen().__aiter__(), self.E)
        return True  # 収集成功としてカウント


# ========= main =========
async def main():
    logging.basicConfig(
        level=LOGLEVEL, format="%(asctime)s %(levelname)s: %(message)s"
    )
    async with aiohttp.ClientSession() as sess:
        api = BlueskyAPI(sess)
        estore = EmbeddingStore()
        gstore = GraphStore()
        crawler = Crawler(api, estore, gstore)
        await crawler.seed(SEED_HANDLES)
        await crawler.run()


if __name__ == "__main__":
    asyncio.run(main())