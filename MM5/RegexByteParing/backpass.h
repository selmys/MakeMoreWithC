#ifndef BACKPASS_H
#define BACKPASS_H

float *getdLdYp(float *probs,int *Y,int lenAlpha);
float *getB2grads(float *dLdYp,int lenAlpha);
float *getW2grads(float *Ha,float *dLdYp,int lenAlpha);
float *getBiasGrads(float *dldz);
float *getGainGrads(float *dldz,float *H);
float *getdhdz(float *H);
float *getdldHn(float *dldz,float *bnGain);
float *getB1grads(float *dldHo);
float *getW1grads(float *dldh,float *emb);
float *getCgrads(float *embGrads,int *X,int lenAlpha);
void update(float *A, float *B, int rows, int cols, float learningRate);

#endif