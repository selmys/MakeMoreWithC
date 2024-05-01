#!/usr/bin/python
import torch
import torch.nn.functional as F

from ctypes import CDLL
from ctypes.util import find_library
libc = CDLL(find_library("c"))

libc.srand(42) # for getting random batches

from pygsl import rng     
#for generating random parameters [C, W1, b1, W2, b2]
r=rng.rng()     

torch.set_printoptions(profile="full")
torch.set_printoptions(precision="6")

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
blocksize = 3;
dimensions = 2;

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

b1 = makeRandom(r,1,100)
W1 = makeRandom(r,blocksize*dimensions,100) / (blocksize*dimensions)**0.5
W2 = makeRandom(r,100,27) * 0.1
b2 = makeRandom(r,1,27)
C = makeRandom(r,27,dimensions)

parameters = [C, W1, b1, W2, b2]
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
    H = torch.tanh(emb.view(-1,blocksize*dimensions) @ W1 + b1)
    logits = H @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    #probs = F.softmax(logits)
    if i%10000 == 0:
        print(loss.data)

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    #update
    for p in parameters:
        p.data += learningRate * p.grad
print('Final loss after training is ',loss)

emb = C[Xtr] # training 
h = torch.tanh(emb.view(-1,blocksize*dimensions) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr)
print('train loss is ',loss.item())

emb = C[Xdev] # development
h = torch.tanh(emb.view(-1,blocksize*dimensions) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print('dev loss is ',loss.item())

# make more names
block_size = 3
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        vv = torch.flatten(probs)
        vvv = vv.tolist()
        n = max(r.multinomial(1,vvv,1)) # n is array of all 26 zeros and 1 one
        ix = n.argmax()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))
