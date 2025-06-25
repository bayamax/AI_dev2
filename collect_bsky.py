#!/usr/bin/env python
"""
collect_bsky.py  (Python ≥3.10, vast.ai などの Linux 環境向け)

目的
-----
Bluesky (AT Protocol) から
  • 各アカウント 10 〜 40 件の投稿
  • 各アカウント 10 〜 40 件の「いいね」対象投稿
  • 各アカウント 20 〜 ∞ 件の follow edge
を 50 GB 以内で取得し、OpenAI text‑embedding‑3‑large (1536 dim, fp32) へ
即時埋め込みして保存する。  
メタは SQLite、ベクトル本体は mmap `.npy` を採用。

実行
-----
$ pip install aiohttp openai atproto numpy torch tqdm
$ export OPENAI_API_KEY=sk-...
$ python collect_bsky.py

途中終了しても再実行で続きから再開。
"""

import asyncio, os, sys, sqlite3, json, logging, random, time
from collections import deque, defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterable

import aiohttp, numpy as np, openai
from tqdm.asyncio import tqdm_asyncio

# ---------- 設定 ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or sys.exit("OPENAI_API_KEY 未設定")
BASE          = "https://public.api.bsky.app/xrpc"
EMB_MODEL     = "text-embedding-3-large"
EMB_DIM       = 1536
BATCH_EMB     = 96                      # OpenAI rate に合わせ調整
MIN_POSTS     = 10
MIN_LIKES     = 10
MIN_FOLLOWS   = 20
MAX_POSTS_ACC = 40                      # 1 アカウントあたりの上限
MAX_ACCOUNTS  = 55_000                  # 打ち切り
OUT_DIR       = Path("dataset")
OUT_DIR.mkdir(exist_ok=True)
LOGLEVEL      = logging.INFO
# ------------------------------------


# ============ I/O ============

class EmbeddingStore:
    """1536‑dim fp32 ベクトルをメモリマップで連続保存し、メタは SQLite"""
    def __init__(self, path: Path = OUT_DIR / "embeddings.npy"):
        self.db = sqlite3.connect(OUT_DIR / "meta.db")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS post(
            idx INTEGER PRIMARY KEY,
            account TEXT,
            post_uri TEXT,
            kind TEXT
          );
          CREATE UNIQUE INDEX IF NOT EXISTS post_uq ON post(account,post_uri,kind);
        """)
        self.arr = self._init_mmap(path)
        self.next = (self.db.execute("SELECT MAX(idx) FROM post").fetchone()[0] or -1) + 1

    def _init_mmap(self, path: Path):
        nrows = 2_500_000            # 2.5M * 1536 * 4 ≒ 14 GB
        if not path.exists():
            np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                      shape=(nrows, EMB_DIM))[:] = 0.
        return np.lib.format.open_memmap(path, mode="r+", dtype=np.float32)

    def add(self, metas: List[Tuple[str, str, str]], vecs: np.ndarray):
        n = len(metas)
        if self.next + n > self.arr.shape[0]:
            raise RuntimeError("mmap full ‑‑ 拡張してください")
        self.arr[self.next:self.next + n] = vecs
        self.db.executemany(
            "INSERT OR IGNORE INTO post(idx,account,post_uri,kind) VALUES (?,?,?,?)",
            [(self.next + i, *m) for i, m in enumerate(metas)]
        )
        self.db.commit()
        self.next += n


class GraphStore:
    """フォローグラフを SQLite に保存"""
    def __init__(self):
        self.db = sqlite3.connect(OUT_DIR / "graph.db")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS follow(
            src TEXT,
            dst TEXT,
            UNIQUE(src,dst)
          );
          CREATE INDEX IF NOT EXISTS idx_src ON follow(src);
        """)

    def add(self, src: str, dsts: Iterable[str]):
        self.db.executemany(
            "INSERT OR IGNORE INTO follow(src,dst) VALUES(?,?)",
            [(src, d) for d in dsts]
        )
        self.db.commit()


# ============ Bluesky API ============

class BlueskyAPI:
    """必要最小限の GET を aiohttp で叩く"""
    def __init__(self, session: aiohttp.ClientSession):
        self.sess = session

    async def _get(self, endpoint: str, **params) -> Dict[str, Any]:
        url = f"{BASE}/{endpoint}"
        while True:
            r = await self.sess.get(url, params=params, timeout=30)
            if r.status in (502, 503, 504):
                await asyncio.sleep(5); continue           # gateway error → リトライ
            if r.status == 429:
                await asyncio.sleep(15); continue          # rate‑limit
            r.raise_for_status()
            return await r.json()

    # プロフィール (did/handle)
    async def profile(self, actor: str) -> Dict[str, Any]:
        return await self._get("app.bsky.actor.getProfile", actor=actor)

    # 投稿
    async def posts(self, actor: str, limit=MAX_POSTS_ACC):
        js = await self._get("app.bsky.feed.getAuthorFeed", actor=actor, limit=limit)
        for item in js.get("feed", []):
            post = item["post"]["record"]
            yield item["post"]["uri"], post.get("text", "")

    # フォロー
    async def follows(self, actor: str, limit=1000):
        js = await self._get("app.bsky.graph.getFollows", actor=actor, limit=limit)
        return [f["handle"] for f in js.get("follows", [])]

    # いいね -> liked post URIs
    async def like_uris(self, actor: str, limit=MAX_POSTS_ACC) -> List[str]:
        js = await self._get("com.atproto.repo.listRecords",
                             repo=actor, collection="app.bsky.feed.like",
                             limit=limit)
        return [rec["value"]["subject"]["uri"] for rec in js.get("records", [])]

    # 複数 URI → posts
    async def posts_by_uri(self, uris: List[str]) -> Dict[str, str]:
        out = {}
        for chunk in [uris[i:i+25] for i in range(0, len(uris), 25)]:
            js = await self._get("app.bsky.feed.getPosts", **{ "uris": chunk })
            for post in js.get("posts", []):
                out[post["uri"]] = post["record"].get("text", "")
        return out


# ============ Embedding ============

openai.api_key = OPENAI_API_KEY

async def embed_texts(texts: List[str]) -> np.ndarray:
    for _ in range(3):
        try:
            res = await openai.embeddings.async_create(model=EMB_MODEL, input=texts)
            return np.asarray([d["embedding"] for d in res.data], dtype=np.float32)
        except openai.RateLimitError:
            await asyncio.sleep(5)
    raise RuntimeError("OpenAI embed failed 3×")


async def embed_stream(stream, store: EmbeddingStore):
    """stream: yield (meta_tuple, text)"""
    buf_meta, buf_txt = [], []
    async for meta, txt in stream:
        if not txt:                      # 空文字
            continue
        buf_meta.append(meta); buf_txt.append(txt[:2048])   # 文字数制限
        if len(buf_txt) >= BATCH_EMB:
            store.add(buf_meta, await embed_texts(buf_txt))
            buf_meta, buf_txt = [], []
    if buf_txt:
        store.add(buf_meta, await embed_texts(buf_txt))


# ============ Crawler ============

class Crawler:
    def __init__(self, api: BlueskyAPI, estore: EmbeddingStore, gstore: GraphStore):
        self.api, self.E, self.G = api, estore, gstore
        self.seen_accounts: set[str] = set()
        self.q = deque()                 # BFS キュー

    async def seed(self, handles: List[str]):
        self.q.extend(handles)

    async def run(self):
        pbar = tqdm_asyncio(total=MAX_ACCOUNTS, desc="accounts", unit="")
        while self.q and len(self.seen_accounts) < MAX_ACCOUNTS:
            actor = self.q.popleft()
            if actor in self.seen_accounts: continue
            try:
                ok = await self.process_actor(actor)
                if ok:
                    self.seen_accounts.add(actor)
                    pbar.update(1)
            except Exception as e:
                logging.warning(f"{actor}: {e}")

    async def process_actor(self, actor: str) -> bool:
        posts = [p async for p in self.api.posts(actor)]
        if len(posts) < MIN_POSTS: return False

        likes_uri = await self.api.like_uris(actor, limit=MAX_POSTS_ACC)
        if len(likes_uri) < MIN_LIKES: return False
        # liked post の本文取得
        like_posts = await self.api.posts_by_uri(likes_uri[:MAX_POSTS_ACC])

        follows = await self.api.follows(actor)
        if len(follows) < MIN_FOLLOWS: return False

        # --- 保存 ---
        async def gen():
            for uri, text in posts[:MAX_POSTS_ACC]:
                yield (actor, uri, "post"), text
            for uri, text in like_posts.items():
                yield (actor, uri, "like"), text
        await embed_stream(gen().__aiter__(), self.E)
        self.G.add(actor, follows)

        # --- 探索 ---
        random.shuffle(follows)
        self.q.extend(follows[:50])
        return True


# ============ main ============

async def main(seed_users: List[str]):
    logging.basicConfig(level=LOGLEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    async with aiohttp.ClientSession() as sess:
        api     = BlueskyAPI(sess)
        estore  = EmbeddingStore()
        gstore  = GraphStore()
        crawler = Crawler(api, estore, gstore)
        await crawler.seed(seed_users)
        await crawler.run()

if __name__ == "__main__":
    # 既知の英語アカウントを種にすると英語圏を中心に拡散しやすい
    SEED = ["bsky.app", "atproto.com", "jay.bsky.social"]
    asyncio.run(main(SEED))