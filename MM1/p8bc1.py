#!/usr/bin/python
import torch
# next 2 lines needed for makemore multinomial
import numpy
from pygsl import rng

words = open('names.txt', 'r').read().splitlines()

N = torch.zeros((27,27), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in words:
	chs = ['.'] + list(w) + ['.']
	for ch1, ch2 in zip(chs, chs[1:]):
		ix1 = stoi[ch1]
		ix2 = stoi[ch2]
		N[ix1, ix2] += 1 # adding all bigrams up


P = (N+1).float() # change counts to floats and smooth
P = P / P.sum(1, keepdim=True) # probabilities --> sum of each row == 1.0
#g = torch.Generator().manual_seed(2147483647)
r = rng.ran0()
for i in range(20):
	out =[]
	ix = 0
	while True:
		p = P[ix]
		#ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
		vv = torch.flatten(p)
		vvv = vv.tolist()
		n = max(r.multinomial(1,vvv,1)) # n is array of all 26 zeros and 1 one
		ix = n.argmax()
		out.append(itos[ix])
		if ix == 0:
			break
	print(''.join(out))
		
# loss is product of all probabilities
# or the addition of the logs of all probabilities
# for each bigram
# known a the log_likelihood
log_likelihood = 0.0
n = 0

for w in words:
	chs = ['.'] + list(w) + ['.']
	for ch1,ch2 in zip(chs, chs[1:]):
		ix1 = stoi[ch1]
		ix2 = stoi[ch2]
		prob = P[ix1, ix2]
		logprob = torch.log(prob)
		log_likelihood += logprob
		n += 1
nll = -log_likelihood/n
print(nll)
