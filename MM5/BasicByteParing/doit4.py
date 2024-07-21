#!/usr/bin/python
import torch
import torch.nn.functional as F
import numpy as np

from ctypes import CDLL
from ctypes.util import find_library
libc = CDLL(find_library("c"))

libc.srand(42) # for getting random batches

from pygsl import rng     
#for generating random parameters [C, W1, b1, W2, b2]
r=rng.rng()     

torch.set_printoptions(profile="full")
torch.set_printoptions(precision="5")

def makeRandom(r,rows,cols):
    # for generating random parameters [C, W1, b1, W2, b2]
    v = []
    for i in range(rows*cols):
        v.append(r.gaussian(1.0))
    v = torch.tensor(v)
    return v.view(rows,cols)

def build_dataset(words):
    block_size = 3
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

words = open('names.txt','r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
block_size = 3;
momentum = 0.1

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

C = makeRandom(r,27,10)
b1 = makeRandom(r,1,200)
W1 = makeRandom(r,30,200) 
W1 /= (30**0.5)
W2 = makeRandom(r,200,27) 
W2 *= 0.1
b2 = makeRandom(r,1,27)
bngain = torch.ones((1, 200))
bnbias = torch.zeros((1, 200))

running_mean = torch.zeros(200)
running_var = torch.ones(200)

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
sum1 = sum(p.nelement() for p in parameters)
print('Number of parameters = ',sum1)
for p in parameters:
    p.requires_grad = True
learningRate = -0.1

for i in range(200000):
    if i >= 100000:
        learningRate = -0.01

    # minibatch construct
    aList = []
    for j in range(32):
        aList.append(libc.rand() % Xtr.shape[0])
    ix = torch.tensor(aList)

    # forward pass
    emb = C[Xtr[ix]] 
    Ho = emb.view(-1,30) @ W1 + b1
    Ho.retain_grad()
    Hn = (Ho - Ho.mean(0, keepdim=True)) / Ho.std(0, keepdim=True)
    running_mean = (1.0 - momentum)*running_mean + momentum*Ho.mean(0, keepdim=True)
    running_var  = (1.0 - momentum)*running_var  + momentum*Ho.std(0, keepdim=True)
    Ha = torch.tanh(bngain * Hn + bnbias)
    logits = Ha @ W2 + b2
    probs = F.softmax(logits,dim=1)
    loss = F.cross_entropy(logits, Ytr[ix])
    #probs = F.softmax(logits)
    if i%10000 == 0:
        print(i,loss.data)

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    #print(Ho.grad) # 32x200
    print(torch.sum(Ho.grad,0)) # 200
    print(b1.grad) # 200 sum of Ho.grad
    quit()
    #update
    for p in parameters:
        p.data += learningRate * p.grad
print('Final loss after training is ',loss)

emb = C[Xtr] # training 
h = emb.view(-1,30) @ W1 + b1
h = (h - h.mean(0, keepdim=True)) / h.std(0, keepdim=True)
h = torch.tanh(bngain * h + bnbias)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr)
print('train loss is ',loss.item())

emb = C[Xdev] # development
h = emb.view(-1,30) @ W1 + b1
h = (h - h.mean(0, keepdim=True)) / h.std(0, keepdim=True)
h = torch.tanh(bngain * h + bnbias)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print('dev loss is ',loss.item())

# make more names
g = torch.Generator().manual_seed(2147483647)
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = emb.view(1, -1) @ W1 + b1
        hn = (h - running_mean) / running_var
        hn = torch.tanh(bngain * hn + bnbias)
        logits = hn @ W2 + b2
        probs = F.softmax(logits, dim=1)
        #v = torch.gather(probs.detach(),1,Ytr[ix].view(-1,1))
        vv = torch.flatten(probs)
        vvv = vv.tolist()
        n = max(r.multinomial(1,vvv,1))
        ix = n.argmax()
        #ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))
