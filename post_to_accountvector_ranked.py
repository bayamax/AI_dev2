#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_to_accountvector_ranked.py
────────────────────────────────────────────────────────
● Hard Negative + Easy Negative (ランダム負例 20 %)
● BPR Loss でランキング学習
● list/tuple 混在バグを根本解決
"""

from pathlib import Path
import os, csv, time, random, traceback
from collections import defaultdict
import numpy as np
from tqdm import tqdm

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ────── Dir & Hyper ───────────────────────────────
CODE_DIR = Path(__file__).resolve().parent
BASE_DIR = CODE_DIR.parent / 'vast'

POSTS_NPY  = BASE_DIR / 'aggregated_posting_vectors.npy'
EDGES_CSV  = BASE_DIR / 'edges.csv'

CACHE_DIR  = BASE_DIR / 'cache'; CACHE_DIR.mkdir(exist_ok=True)
TRAIN_EDGES = CACHE_DIR / 'train_edges.npy'
VAL_EDGES   = CACHE_DIR / 'val_edges.npy'

OUT_DIR = BASE_DIR / 'attention_rank_out'; OUT_DIR.mkdir(exist_ok=True)
LOG_FILE   = OUT_DIR / 'rank_training.log'
MODEL_PATH = OUT_DIR / 'rank_follow_model.pt'

POST_DIM, ACCOUNT_DIM, HIDDEN_DIM = 3072, 128, 256
MAX_POSTS, BATCH_SIZE             = 50, 256
NUM_EPOCHS, LR, WD, EARLY_STOP    = 50, 3e-4, 1e-5, 8

NEG_PER_POS   = 3      # HardNeg : 正例1つにつき 3 個
HARD_TOPN     = 20     # HardNeg 候補は類似度上位 20
EASY_NEG_RATE = 0.2    # EasyNeg : 正例の 20 %

VALID_RATIO   = 0.15
# ────────────────────────────────────────────────

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}  {msg}")
    with LOG_FILE.open('a') as f: f.write(f"{ts}  {msg}\n")

# ---- posts ----
def load_posts():
    return np.load(POSTS_NPY, allow_pickle=True).item()

# ---- edges (tuple 化でバグ根絶) ----
def load_split_edges():
    if TRAIN_EDGES.exists():                      # キャッシュ読み込み
        tr = [tuple(x) for x in np.load(TRAIN_EDGES, allow_pickle=True)]
        va = [tuple(x) for x in np.load(VAL_EDGES,   allow_pickle=True)]
        return tr, va

    # CSV → 分割
    with EDGES_CSV.open() as f:
        rdr = csv.reader(f); next(rdr)
        edges = [tuple(row[:2]) for row in rdr]
    random.shuffle(edges)
    k = int(len(edges)*(1-VALID_RATIO))
    np.save(TRAIN_EDGES, np.array(edges[:k], dtype=object), allow_pickle=True)
    np.save(VAL_EDGES,   np.array(edges[k:],  dtype=object), allow_pickle=True)
    return edges[:k], edges[k:]

# ---- model ----
class AttentionPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(POST_DIM, 8, batch_first=True)
        self.ln  = nn.LayerNorm(POST_DIM)
        self.proj= nn.Sequential(nn.Linear(POST_DIM,ACCOUNT_DIM*2),nn.GELU(),
                                 nn.Linear(ACCOUNT_DIM*2,ACCOUNT_DIM))
        self.sc  = nn.Linear(POST_DIM,1,bias=False)
    def forward(self,x,m):
        x = F.normalize(x,p=2,dim=-1)
        x = self.ln(x + self.mha(x,x,x,key_padding_mask=(m==0))[0])
        w = F.softmax(self.sc(x).squeeze(-1).masked_fill(m==0,-1e9),-1)
        v = (x*w.unsqueeze(-1)).sum(1)
        return F.normalize(self.proj(v),p=2,dim=-1)

class FollowPred(nn.Module):
    def __init__(self):
        super().__init__(); h=HIDDEN_DIM
        self.fe = nn.Sequential(nn.Linear(ACCOUNT_DIM,h),nn.GELU())
        self.te = nn.Sequential(nn.Linear(ACCOUNT_DIM,h),nn.GELU())
        self.head = nn.Sequential(nn.Linear(h*3,h),nn.GELU(),
                                  nn.Linear(h,h//2),nn.GELU(),
                                  nn.Linear(h//2,1))
    def forward(self,f,t):
        f,t=self.fe(f),self.te(t)
        return self.head(torch.cat([f,t,f*t],-1)).squeeze(-1)

class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.ap=AttentionPool(); self.fp=FollowPred()
    def score(self,fp,fm,tp,tm):
        return self.fp(self.ap(fp,fm), self.ap(tp,tm))

# ---- utils ----
def pad_posts(lst):
    arr = np.zeros((MAX_POSTS, POST_DIM), np.float32)
    msk = np.zeros(MAX_POSTS, np.float32)
    L = min(len(lst), MAX_POSTS)
    if L: arr[:L] = lst[:L]; msk[:L] = 1
    return arr, msk

def compute_acc_vecs(posts, device, ap):
    users = list(posts.keys()); acc={}
    ap.eval()
    with torch.no_grad():
        for i in range(0,len(users),512):
            fp,fm = zip(*(pad_posts(posts[u]) for u in users[i:i+512]))
            fp = torch.tensor(np.array(fp), device=device)
            fm = torch.tensor(np.array(fm), device=device)
            v  = ap(fp, fm).cpu().numpy()
            acc.update({u:vec for u,vec in zip(users[i:i+512],v)})
    return acc

def hard_negs(acc_vecs, pos_edges):
    edge_set = set(pos_edges)
    users = list(acc_vecs.keys())
    M = torch.tensor([acc_vecs[u] for u in users])
    M = F.normalize(M,dim=1)
    sims = M @ M.T
    neg=[]
    for f,t in pos_edges:
        i = users.index(f)
        vals,idx = sims[i].topk(HARD_TOPN+1)
        cands=[users[j] for j in idx[1:].tolist() if (f,users[j]) not in edge_set]
        random.shuffle(cands)
        neg.extend([(f,u) for u in cands[:NEG_PER_POS]])
    return neg

def easy_negs(posts, pos_edges):
    cnt = int(len(pos_edges)*EASY_NEG_RATE)
    users = list(posts.keys())
    edge_set = set(pos_edges)
    neg=[]
    while len(neg)<cnt:
        f,t = random.sample(users,2)
        if (f,t) not in edge_set:
            neg.append((f,t))
    return neg

# ---- Dataset (triplet) ----
class BPRDataset(Dataset):
    def __init__(self,pos_edges,neg_edges,posts):
        self.posts = posts
        self.trip  = []
        valid={u for u,p in posts.items() if p}
        neg_by_f = defaultdict(list)
        for f,n in neg_edges:
            if n in valid: neg_by_f[f].append(n)
        for f,p in pos_edges:
            if f in valid and p in valid and neg_by_f[f]:
                for n in neg_by_f[f]:
                    self.trip.append((f,p,n))
    def __len__(self): return len(self.trip)
    def __getitem__(self,idx):
        f,p,n = self.trip[idx]
        fp,fm = pad_posts(self.posts[f])
        tp,tm = pad_posts(self.posts[p])
        np_,nm= pad_posts(self.posts[n])
        return (torch.tensor(fp),torch.tensor(fm),
                torch.tensor(tp),torch.tensor(tm),
                torch.tensor(np_),torch.tensor(nm))

# ---- Train ----
def train():
    dev=torch.device('cuda' if torch.cuda.is_available()
                     else 'mps' if torch.backends.mps.is_available()
                     else 'cpu')
    log(f"device={dev}")
    posts      = load_posts()
    train_e,val_e = load_split_edges()
    model      = Model().to(dev)
    opt        = torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)

    best,wait=1e9,0
    for ep in range(1,NUM_EPOCHS+1):
        acc_vecs = compute_acc_vecs(posts,dev,model.ap)
        hard = hard_negs(acc_vecs, train_e)
        easy = easy_negs(posts,    train_e)
        neg_e=hard+easy

        ds = BPRDataset(train_e, neg_e, posts)
        dl = DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)

        model.train(); tot=0
        for fp,fm,tp,tm,np_,nm in tqdm(dl,desc=f"E{ep}"):
            fp,fm,tp,tm,np_,nm=[x.to(dev) for x in (fp,fm,tp,tm,np_,nm)]
            pos = model.score(fp,fm,tp,tm)
            neg = model.score(fp,fm,np_,nm)
            loss = -torch.log(torch.sigmoid(pos-neg)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*len(pos)
        avg=tot/len(ds); log(f"[{ep}] loss={avg:.4f}")

        if avg < best:
            best,wait=avg,0
            torch.save(model.state_dict(), MODEL_PATH)
            log("  saved best")
        else:
            wait+=1
            if wait>=EARLY_STOP:
                log("early stop"); break

if __name__=="__main__":
    try: train()
    except Exception as e:
        log(f"Error: {e}"); log(traceback.format_exc())