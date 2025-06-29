#!/usr/bin/env python
"""
collect_bsky_no_like.py  –  Bluesky 収集 (投稿+フォローのみ版)
  • Like コレクションは呼ばない
  • App Password 認証は投稿・フォロー取得だけに使う
  • 投稿10件 / フォロー5件 を満たせば保存
"""

from __future__ import annotations
import asyncio, logging, os, random, sqlite3, sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import aiohttp, numpy as np, openai
from aiohttp import ClientResponseError
from atproto import Client as BskyClient
from tqdm.asyncio import tqdm_asyncio

# ───── 環境変数 ─────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")        or sys.exit("OPENAI_API_KEY 未設定")
BSKY_IDENTIFIER  = os.getenv("BSKY_IDENTIFIER")       # 例: foo.bsky.social
BSKY_APP_PASSWORD= os.getenv("BSKY_APP_PASSWORD")     # 例: xxxx-xxxx-xxxx-xxxx
# ────────────────────

SEED_HANDLES = [
    # 実在ユーザを複数指定すると探索が途切れにくい
    "ngntrtr.bsky.social",
    "uishig.bsky.social",
    "mikapikazompz.bsky.social",
    "purinharumaki.bsky.social",
    "sora-sakurai.bsky.social",
]
BASE_ENDPOINT = "https://public.api.bsky.app/xrpc"
EMB_MODEL, EMB_DIM, BATCH_EMB = "text-embedding-3-large", 1536, 96
MIN_POSTS, MIN_FOLLOWS = 10, 5           # Like 条件を外した
MAX_POSTS_ACC, MAX_ACCOUNTS = 40, 55_000
OUT_DIR = Path("dataset"); OUT_DIR.mkdir(exist_ok=True)
LOGLEVEL = logging.INFO
# ────────────────────


# ========= EmbeddingStore =========
class EmbeddingStore:
    def __init__(self, n_rows: int = 2_000_000):
        self.db = sqlite3.connect(OUT_DIR / "meta.db")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS post(
              idx INTEGER PRIMARY KEY,
              account TEXT,
              post_uri TEXT,
              kind TEXT,
              UNIQUE(account,post_uri,kind));
        """)
        path = OUT_DIR / "embeddings.npy"
        if not path.exists():
            np.lib.format.open_memmap(path, "w+", np.float32,
                                      shape=(n_rows, EMB_DIM))[:] = 0.
        self.arr = np.lib.format.open_memmap(path, "r+", np.float32)
        self.next = (self.db.execute("SELECT MAX(idx) FROM post").fetchone()[0] or -1) + 1

    def add(self, metas: List[Tuple[str,str,str]], vecs: np.ndarray):
        n = len(metas)
        if self.next + n > self.arr.shape[0]:
            raise RuntimeError("mmap full – 拡張してください")
        self.arr[self.next:self.next+n] = vecs
        self.db.executemany(
            "INSERT OR IGNORE INTO post(idx,account,post_uri,kind) VALUES (?,?,?,?)",
            [(self.next+i,*m) for i,m in enumerate(metas)])
        self.db.commit(); self.next += n


# ========= GraphStore =========
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
            [(src,d) for d in dsts]); self.db.commit()


# ========= Bluesky API =========
class BlueskyAPI:
    def __init__(self, sess:aiohttp.ClientSession):
        self.sess = sess
        self._did_cache: Dict[str,str] = {}

        if BSKY_IDENTIFIER and BSKY_APP_PASSWORD:
            cl = BskyClient(); cl.login(BSKY_IDENTIFIER, BSKY_APP_PASSWORD)
            self.jwt = (
                getattr(cl, "access_jwt", None)
                or getattr(getattr(cl, "current_session", None) or object(), "accessJwt", None)
                or getattr(getattr(cl, "session", None) or object(), "access_jwt", None)
                or getattr(getattr(cl, "_session", None) or object(), "access_jwt", None)
            )
            logging.info(f"logged-in as {BSKY_IDENTIFIER}")
        else:
            self.jwt = None
            logging.info("認証なし (公開データのみ取得)")

    async def _get(self, ep:str, **params)->Dict[str,Any]:
        url=f"{BASE_ENDPOINT}/{ep}"
        headers={"Accept":"application/json"}
        if self.jwt: headers["Authorization"]=f"Bearer {self.jwt}"
        while True:
            r=await self.sess.get(url, params=params, headers=headers, timeout=30)
            if r.status in (502,503,504): await asyncio.sleep(5); continue
            if r.status==429: await asyncio.sleep(15); continue
            r.raise_for_status(); return await r.json()

    async def _did(self, handle:str)->str:
        if handle in self._did_cache: return self._did_cache[handle]
        js=await self._get("app.bsky.actor.getProfile", actor=handle)
        did=js["did"].strip(); self._did_cache[handle]=did; return did

    async def posts(self, actor:str, limit:int=MAX_POSTS_ACC):
        js=await self._get("app.bsky.feed.getAuthorFeed",
                           actor=actor, limit=min(limit,100))
        for it in js.get("feed", []):
            yield it["post"]["uri"], it["post"]["record"].get("text","")

    async def follows(self, actor:str, limit:int=100)->List[str]:
        try:
            js=await self._get("app.bsky.graph.getFollows",
                               actor=actor, limit=min(limit,100))
        except ClientResponseError as e:
            if e.status in (400,404): return []
            raise
        return [u["handle"] for u in js.get("follows", [])]


# ========= Embedding =========
openai.api_key = OPENAI_API_KEY
async def embed_texts(texts:List[str])->np.ndarray:
    for _ in range(3):
        try:
            r=await openai.embeddings.async_create(model=EMB_MODEL, input=texts)
            return np.asarray([d["embedding"] for d in r.data], np.float32)
        except openai.RateLimitError: await asyncio.sleep(5)
    raise RuntimeError("embedding failed")

async def embed_stream(stream, store:EmbeddingStore):
    buf_meta, buf_txt = [], []
    async for meta,txt in stream:
        if not txt: continue
        buf_meta.append(meta); buf_txt.append(txt[:2048])
        if len(buf_txt)>=BATCH_EMB:
            store.add(buf_meta, await embed_texts(buf_txt))
            buf_meta, buf_txt=[],[]
    if buf_txt: store.add(buf_meta, await embed_texts(buf_txt))


# ========= Crawler =========
class Crawler:
    def __init__(self, api:BlueskyAPI, estore:EmbeddingStore, gstore:GraphStore):
        self.api, self.E, self.G = api, estore, gstore
        self.seen:set[str]=set(); self.q:deque[str]=deque()

    async def seed(self, handles:List[str]): self.q.extend(handles)

    async def run(self):
        vis=tqdm_asyncio(total=MAX_ACCOUNTS, desc="visited"); sto=tqdm_asyncio(total=MAX_ACCOUNTS, desc="stored")
        while self.q and len(self.seen)<MAX_ACCOUNTS:
            actor=self.q.popleft()
            if actor in self.seen: continue
            self.seen.add(actor); vis.update(1)
            try:
                ok=await self.process_actor(actor)
                if ok: sto.update(1)
            except Exception as e:
                logging.warning(f"{actor}: {e}")

    async def process_actor(self, actor:str)->bool:
        posts=[p async for p in self.api.posts(actor)]
        follows=await self.api.follows(actor)

        logging.debug(f"{actor} posts={len(posts)} follows={len(follows)}")

        self.G.add(actor,follows)
        random.shuffle(follows); self.q.extend(follows[:50])

        if len(posts)<MIN_POSTS or len(follows)<MIN_FOLLOWS:
            return False

        async def gen():
            for uri,txt in posts[:MAX_POSTS_ACC]:
                yield (actor,uri,"post"), txt
        await embed_stream(gen().__aiter__(), self.E)
        return True


# ========= main =========
async def main():
    logging.basicConfig(level=LOGLEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    async with aiohttp.ClientSession() as sess:
        api=BlueskyAPI(sess); estore=EmbeddingStore(); gstore=GraphStore()
        crawler=Crawler(api,estore,gstore)
        await crawler.seed(SEED_HANDLES); await crawler.run()

if __name__=="__main__":
    asyncio.run(main())