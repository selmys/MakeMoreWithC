#ifndef NORMALIZE_H
#define NORMALIZE_H

float *getHmean(float *H);
float *getHmean1(float *Ho, float *Mu);
float *getHvar(float *H, float *mean);
float *getHdev(float *hVar);
float *getHnormal(float *Ho,float *hMean,float *sDev,int batchSize);
float *normalizeH(float *H,float *r_mean,float *r_var,int testing,int batchSize);

#endif