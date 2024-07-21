#ifndef TESTING_H
#define TESTING_H

double *probs2d(float *probs,int lenAlpha);
int *makeMore(float *C,float *W1,float *B1,float *W2,float *B2,
                int count, float *bnGain,float *bnBias,
                float *running_mean, float *running_var,
                int lenAlpha,int numRawBytes);

#endif