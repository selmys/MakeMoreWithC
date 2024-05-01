#ifndef TESTING_H
#define TESTING_H

double *probs2d(float *probs);
void makeMore(char *alphabet,float *C,float *W1,float *B1,float *W2,float *B2,int count,
                float *bnGain,float *bnBias,float *running_mean,float *running_var);
float splitLoss(char *names,float *C,float *W1,float *B1,float *W2,float *B2,char *alphabet,
                float *bnGain,float *bnBias,float *running_mean,float *running_var);                

#endif