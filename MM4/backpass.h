#ifndef BACKPASS_H
#define BACKPASS_H

float *getdLdY(float *Ha,float *probs,int *Y,float *W2);
float *getdLdYp(float *probs,int *Y);
float *getB2grads(float *dLdYp);
float *getW2grads(float *Ha,float *dLdYp);
float *getBiasGrads(float *dldz);
float *getGainGrads(float *dldz,float *H);
float *getdhdz(float *H);
float *getdldHn(float *dldz,float *bnGain);
float *getB1grads(float *dldHo);
float *getW1grads(float *dldh,float *emb);
float *getCgrads(float *embGrads,int *X);
void update(float *A, float *B, int rows, int cols, float learningRate);

#endif