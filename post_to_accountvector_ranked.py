#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_to_accountvector_ranked.py
────────────────────────────────────────
BPR (Bayesian Personalized Ranking) で
「正例 > 負例」になるよう学習する版
"""

from pathlib import Path
import os, csv, time, random, traceback
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ───── Dir & Hyper ─────
CODE_DIR = Path(__file__).resolve().parent
BASE_DIR = CODE_DIR.parent / 'vast'

POSTS_NPY  = BASE_DIR / 'aggregated_posting_vectors.npy'
EDGES_CSV  = BASE_DIR / 'edges.csv'

CACHE_DIR  = BASE_DIR / 'cache';               CACHE_DIR.mkdir(exist_ok=True)
TRAIN_EDGES = CACHE_DIR / 'train_edges.npy'
VAL_EDGES   = CACHE_DIR / 'val_edges.npy'

OUT_DIR  = BASE_DIR / 'attention_rank_out';    OUT_DIR.mkdir(exist_ok=True)
LOG_FILE   = OUT_DIR / 'rank_training.log'
MODEL_PATH = OUT_DIR / 'rank_follow_model.pt'

POST_DIM, ACCOUNT_DIM, HIDDEN_DIM = 3072, 128, 256
MAX_POSTS, BATCH_SIZE = 50, 256
NUM_EPOCHS, LR, WD, EARLY_STOP = 50, 3e-4, 1e-5, 8
NEG_PER_POS = 3               # 正例1つにつき HardNeg を3つ
HARD_TOPN   = 20              # アカベク類似度上位から負例抽出
VALID_RATIO = 0.15
# ──────────────────────

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}  {msg}")
    with LOG_FILE.open('a') as f: f.write(f"{ts}  {msg}\n")

# ---- load posts (npy) ----
def load_posts():
    return np.load(POSTS_NPY, allow_pickle=True).item()

# ---- edge split ----
def load_split_edges():
    if TRAIN_EDGES.exists():
        return (np.load(TRAIN_EDGES, allow_pickle=True).tolist(),
                np.load(VAL_EDGES,   allow_pickle=True).tolist())
    edges = []
    with EDGES_CSV.open() as f:
        rdr = csv.reader(f); next(rdr)
        edges = [(s,t) for s,t,*_ in rdr]
    random.shuffle(edges)
    k = int(len(edges)*(1-VALID_RATIO))
    np.save(TRAIN_EDGES, edges[:k], allow_pickle=True)
    np.save(VAL_EDGES,   edges[k:], allow_pickle=True)
    return edges[:k], edges[k:]

# ---- AttentionPool & FollowPred ----
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
    arr=np.zeros((MAX_POSTS,POST_DIM),np.float32)
    msk=np.zeros(MAX_POSTS,np.float32)
    L=min(len(lst),MAX_POSTS)
    if L: arr[:L]=lst[:L]; msk[:L]=1
    return arr,msk

# ---- Hard Negative maker ----
def compute_acc_vecs(posts, device, model_ap):
    users=list(posts.keys())
    acc_vecs={}
    model_ap.eval()
    with torch.no_grad():
        for i in range(0,len(users),512):
            fp,fm=zip(*(pad_posts(posts[u]) for u in users[i:i+512]))
            v=model_ap(torch.tensor(fp).to(device),
                       torch.tensor(fm).to(device)).cpu().numpy()
            acc_vecs.update({u:vec for u,vec in zip(users[i:i+512],v)})
    return acc_vecs

def hard_negs(acc_vecs, pos_edges, topn=HARD_TOPN, neg_per_pos=NEG_PER_POS):
    edge_set=set(pos_edges)
    users=list(acc_vecs.keys())
    M=torch.tensor([acc_vecs[u] for u in users]); M=F.normalize(M,dim=1)
    sims=M@M.T
    neg=[]
    for (f,t) in pos_edges:
        i,j=users.index(f),users.index(t)
        # 上位 topn 類似ユーザからランダム選択
        vals,idx=sims[i].topk(topn+1)
        cands=[users[k] for k in idx[1:].tolist() if (f,users[k]) not in edge_set]
        random.shuffle(cands)
        for u in cands[:neg_per_pos]:
            neg.append((f,u))
    return neg

# ---- Dataset: 正例+Hard Neg を triplet で返す ----
class BPRDataset(Dataset):
    def __init__(self,pos_edges,neg_edges,posts):
        self.posts=posts
        self.triplets=[]
        valid={u for u,p in posts.items() if p}
        for f,t in pos_edges:
            if f in valid and t in valid:
                negs=[n for (ff,n) in neg_edges if ff==f and n in valid]
                for n in negs:
                    self.triplets.append((f,t,n))
    def __len__(self): return len(self.triplets)
    def __getitem__(self,idx):
        f,p,n=self.triplets[idx]
        fp,fm = pad_posts(self.posts[f])
        tp,tm = pad_posts(self.posts[p])
        np_,nm= pad_posts(self.posts[n])
        return (torch.tensor(fp),torch.tensor(fm),
                torch.tensor(tp),torch.tensor(tm),
                torch.tensor(np_),torch.tensor(nm))

# ---- Train loop ----
def train():
    dev=torch.device('cuda' if torch.cuda.is_available()
                     else 'mps' if torch.backends.mps.is_available()
                     else 'cpu')
    log(f"device={dev}")
    posts=load_posts()
    train_e,val_e=load_split_edges()
    model=Model().to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)

    # Precompute account vecs for HardNeg
    acc_vecs=compute_acc_vecs(posts,dev,model.ap)

    best,wait=1e9,0
    for ep in range(1,NUM_EPOCHS+1):
        neg_e = hard_negs(acc_vecs, train_e)
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

        # rudimentary val metric: AUC-like
        if avg < best:
            best,wait=avg,0
            torch.save(model.state_dict(), MODEL_PATH)
            log("  saved best")
        else:
            wait+=1
            if wait>=EARLY_STOP: log("early stop"); break

if __name__=="__main__":
    try: train()
    except Exception as e:
        log(f"Error: {e}"); log(traceback.format_exc())