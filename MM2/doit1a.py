#!/usr/bin/python
import torch
import torch.nn.functional as F

import numpy
from pygsl import rng
r=rng.rng() 

def makeRandom(r,rows,cols):
    # for generating random parameters
    v = []
    for i in range(rows*cols):
        v.append(r.gaussian(1.0))
    v = torch.tensor(v)
    return v.view(rows,cols)

words = open('names.txt','r').read().splitlines()
chars = sorted(list(set(''.join(words))))
chars = list('.') + chars
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for s,i in stoi.items()}

xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ',num)
#g = torch.Generator().manual_seed(2147483647)
#w = torch.randn((27,27),generator=g, requires_grad=True)
w = makeRandom(r,27,27)
w.requires_grad = True
for x in range(150):
    # forward pass
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ w
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(num), ys].log().mean()# + 0.01*(w**2).mean()
    print('loss = ',loss.item())
    # backward pass
    w.grad = None
    loss.backward()
    w.data = w.data -50 * w.grad
# make more
for i in range(20):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ w
        #counts = logits.exp()
        #p = counts / counts.sum(1, keepdims=True)
        p = F.softmax(logits, dim=1)
        #ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        vv = torch.flatten(p)
        vvv = vv.tolist()
        n = max(r.multinomial(1,vvv,1)) # n is array of all 26 zeros and 1 one
        ix = n.argmax()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
