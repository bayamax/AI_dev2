#!/usr/bin/env python
"""
collect_bsky.py  –  Bluesky 収集スクリプト v0.5
────────────────────────────────────────────────────────
* DEBUG ログ常時 ON
* getPosts URI 連結修正済み
* フォロー API 400/404 → 0 件扱い
"""

from __future__ import annotations
import asyncio, logging, os, random, sqlite3, sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import aiohttp, numpy as np, openai
from aiohttp import ClientResponseError
from tqdm.asyncio import tqdm_asyncio

# ───────── ユーザ設定 ─────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or sys.exit("OPENAI_API_KEY 未設定")

SEED_HANDLES = [
    "jay.bsky.social",
    "sara.bsky.social",
    "blaine",
    "manifold.bsky.social",
    "karlicoss.com",
]

BASE_ENDPOINT = "https://public.api.bsky.app/xrpc"
EMB_MODEL, EMB_DIM, BATCH_EMB = "text-embedding-3-large", 1536, 96

# まずは 1-1-1 で動作確認
MIN_POSTS, MIN_LIKES, MIN_FOLLOWS = 1, 1, 1
MAX_POSTS_ACC, MAX_ACCOUNTS = 40, 55_000

OUT_DIR = Path("dataset"); OUT_DIR.mkdir(exist_ok=True)
LOGLEVEL = logging.DEBUG        # ← 必ず DEBUG 表示
# ──────────────────────────────


# ========= I/O =========
class EmbeddingStore:
    def __init__(self, n_rows: int = 2_500_000):
        self.db = sqlite3.connect(OUT_DIR / "meta.db")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS post(
              idx INTEGER PRIMARY KEY,
              account TEXT,
              post_uri TEXT,
              kind TEXT,
              UNIQUE(account,post_uri,kind));
        """)
        arr_path = OUT_DIR / "embeddings.npy"
        if not arr_path.exists():
            np.lib.format.open_memmap(arr_path, "w+", np.float32,
                                      shape=(n_rows, EMB_DIM))[:] = 0.
        self.arr  = np.lib.format.open_memmap(arr_path, "r+", np.float32)
        self.next = (self.db.execute("SELECT MAX(idx) FROM post").fetchone()[0] or -1) + 1

    def add(self, metas: List[Tuple[str,str,str]], vecs: np.ndarray):
        n = len(metas)
        if self.next + n > self.arr.shape[0]:
            raise RuntimeError("mmap full – 拡張してください")
        self.arr[self.next:self.next+n] = vecs
        self.db.executemany(
            "INSERT OR IGNORE INTO post(idx,account,post_uri,kind) VALUES (?,?,?,?)",
            [(self.next+i,*m) for i,m in enumerate(metas)])
        self.db.commit();  self.next += n


class GraphStore:
    def __init__(self):
        self.db = sqlite3.connect(OUT_DIR / "graph.db")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS follow(
              src TEXT,
              dst TEXT,
              UNIQUE(src,dst));
            CREATE INDEX IF NOT EXISTS idx_src ON follow(src);
        """)
    def add(self, src:str, dsts:Iterable[str]):
        self.db.executemany(
            "INSERT OR IGNORE INTO follow(src,dst) VALUES (?,?)",
            [(src,d) for d in dsts])
        self.db.commit()


# ========= Bluesky API =========
class BlueskyAPI:
    def __init__(self, sess:aiohttp.ClientSession): self.sess=sess

    async def _get(self, ep:str, **params)->Dict[str,Any]:
        url=f"{BASE_ENDPOINT}/{ep}"
        while True:
            r=await self.sess.get(url, params=params, timeout=30)
            if r.status in (502,503,504): await asyncio.sleep(5); continue
            if r.status==429: await asyncio.sleep(15); continue
            r.raise_for_status(); return await r.json()

    async def posts(self, actor:str, limit:int=MAX_POSTS_ACC):
        js=await self._get("app.bsky.feed.getAuthorFeed",
                           actor=actor, limit=min(limit,100))
        for it in js.get("feed", []):
            rec=it["post"]["record"]
            yield it["post"]["uri"], rec.get("text","")

    async def follows(self, actor:str, limit:int=100) -> List[str]:
        try:
            js=await self._get("app.bsky.graph.getFollows",
                               actor=actor, limit=min(limit,100))
        except ClientResponseError as e:
            if e.status in (400,404): return []
            raise
        return [u["handle"] for u in js.get("follows", [])]

    async def like_uris(self, actor:str, limit:int=MAX_POSTS_ACC) -> List[str]:
        try:
            js=await self._get("com.atproto.repo.listRecords",
                               repo=actor, collection="app.bsky.feed.like",
                               limit=min(limit,100))
        except ClientResponseError as e:
            if e.status==404: return []
            raise
        return [rec["value"]["subject"]["uri"] for rec in js.get("records", [])]

    async def posts_by_uri(self, uris:List[str]) -> Dict[str,str]:
        out:Dict[str,str]={}
        for chunk in [uris[i:i+25] for i in range(0,len(uris),25)]:
            joined=",".join(chunk)                    # カンマ区切り
            js=await self._get("app.bsky.feed.getPosts", uris=joined)
            for p in js.get("posts", []):
                txt=p["record"].get("text","")
                if txt: out[p["uri"]] = txt
        return out


# ========= Embedding =========
openai.api_key = OPENAI_API_KEY
async def embed_texts(texts:List[str]) -> np.ndarray:
    for _ in range(3):
        try:
            r=await openai.embeddings.async_create(model=EMB_MODEL, input=texts)
            return np.asarray([d["embedding"] for d in r.data], np.float32)
        except openai.RateLimitError: await asyncio.sleep(5)
    raise RuntimeError("OpenAI embed failed (3 retries)")

async def embed_stream(stream, store:EmbeddingStore):
    buf_meta, buf_txt = [], []
    async for meta,txt in stream:
        if not txt: continue
        buf_meta.append(meta); buf_txt.append(txt[:2048])
        if len(buf_txt)>=BATCH_EMB:
            store.add(buf_meta, await embed_texts(buf_txt))
            buf_meta, buf_txt = [], []
    if buf_txt: store.add(buf_meta, await embed_texts(buf_txt))


# ========= Crawler =========
class Crawler:
    def __init__(self, api:BlueskyAPI, estore:EmbeddingStore, gstore:GraphStore):
        self.api, self.E, self.G = api, estore, gstore
        self.seen:set[str]=set(); self.q:deque[str]=deque()

    async def seed(self, handles:List[str]): self.q.extend(handles)

    async def run(self):
        visited=tqdm_asyncio(total=MAX_ACCOUNTS, desc="visited", unit="")
        stored =tqdm_asyncio(total=MAX_ACCOUNTS, desc="stored ", unit="")
        while self.q and len(self.seen)<MAX_ACCOUNTS:
            actor=self.q.popleft()
            if actor in self.seen: continue
            self.seen.add(actor); visited.update(1)
            try:
                ok=await self.process_actor(actor)
                if ok: stored.update(1)
            except Exception as e:
                logging.warning(f"{actor}: {e}")

    async def process_actor(self, actor:str)->bool:
        logging.debug(f"▼ {actor} 収集中 …")
        posts    =[p async for p in self.api.posts(actor)]
        follows  =await self.api.follows(actor)
        like_uri =await self.api.like_uris(actor)
        like_posts=await self.api.posts_by_uri(like_uri[:MAX_POSTS_ACC])

        logging.debug(
            f"■ {actor} posts={len(posts)} likes_uri={len(like_uri)} "
            f"likes_resolved={len(like_posts)} follows={len(follows)}")

        # グラフ保存・探索
        self.G.add(actor, follows)
        random.shuffle(follows); self.q.extend(follows[:50])

        if (len(posts)<MIN_POSTS or len(like_posts)<MIN_LIKES or len(follows)<MIN_FOLLOWS):
            return False

        async def gen():
            for uri,txt in posts[:MAX_POSTS_ACC]:
                yield (actor,uri,"post"), txt
            for uri,txt in like_posts.items():
                yield (actor,uri,"like"), txt
        await embed_stream(gen().__aiter__(), self.E)
        return True


# ========= main =========
async def main():
    logging.basicConfig(level=LOGLEVEL,
                        format="%(asctime)s %(levelname)s: %(message)s")
    async with aiohttp.ClientSession() as sess:
        api,estore,gstore = BlueskyAPI(sess), EmbeddingStore(), GraphStore()
        crawler=Crawler(api,estore,gstore)
        await crawler.seed(SEED_HANDLES); await crawler.run()

if __name__=="__main__":
    asyncio.run(main())