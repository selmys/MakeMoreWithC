#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>

#include "helpers.h"
#include "testing.h"
#include "normalize.h"
#include "matrix.h"
#include "forepass.h"
#include "backpass.h"
#include "params.h"

double *probs2d(float *probs){
    double *p=(double *)calloc(LENALPHA,sizeof(double));
    for(int i=0;i<LENALPHA;i++)
        p[i]=probs[i];
    return p;
}
void makeMore(char *alphabet,float *C,float *W1,float *B1,float *W2,float *B2,int count,
                float *bnGain,float *bnBias,float *running_mean,float *running_var){
    // let's make count more names
    // but first we need to set up the RNG
    // using GSL (GNU Scientific Library)
    // first select the rng -random number generator- to use
    gsl_rng *pr = gsl_rng_alloc(gsl_rng_ran0);
    // then set the seed to the rng
    gsl_rng_set(pr, 2147483647); // use same as in Andrej's videos
    //gsl_rng_set(pr, time(0));
    int ix=0,k;
    float *H,*logits,*probs,*Hn;
    double *dprobs;
    unsigned int *n = (unsigned int *)calloc(LENALPHA,sizeof(int));
    float *emb=(float *)calloc(INNODES,sizeof(float));
    int numBigrams=1;
    int X[BLOCKSIZE];
    for(int i=0;i<count;i++){
        for(int a=0;a<BLOCKSIZE;a++) X[a]=0;
        while(1){
            //embed X into C giving embedded array (emb)
            embedCX(emb,C,X,numBigrams);
            // compute the hidden layer
            H=multAB(emb,W1,numBigrams,INNODES,INNODES,HIDNODES);
            // add bias to hidden layer
            addBias(H,B1,numBigrams,BIAS1);
            // normalize the hidden layer
            Hn=normalizeH(H,running_mean,running_var,1,numBigrams);
            // multiply hNormal by the Gain
            multMxV(Hn,bnGain,numBigrams,HIDNODES);
            // now add bnBias to hNormXbnGain
            addBias(Hn,bnBias,numBigrams,HIDNODES);
            // use activation function tanh
            addTanh(Hn,numBigrams,HIDNODES);
            // now get the logits (log counts)
            logits=multAB(Hn,W2,numBigrams,HIDNODES,HIDNODES,OUTNODES);
            // add bias to the logits
            addBias(logits,B2,numBigrams,BIAS2);
            // get the probabilities
            probs=softMax(logits,numBigrams,LENALPHA);
            dprobs=probs2d(probs);
            gsl_ran_multinomial(pr,LENALPHA,1,dprobs,n);
            // find which element of n has a 1
            for(int j=0;j<LENALPHA;j++){
                if(n[j]!=0) ix=j;
            }
            if(ix != 0)
                printf("%c",alphabet[ix]);
            else
                break;
            for(k=0;k<BLOCKSIZE-1;k++)
                X[k]=X[k+1];
            X[k]=ix;
        }
        printf(".\n");
    }
    return;
}
float splitLoss(char *names,float *C,float *W1,float *B1,float *W2,float *B2,char *alphabet,
                float *bnGain,float *bnBias,float *running_mean,float *running_var){
    // this function does not update any weights or biases or the lookup table
    int Bigrams = strlen(names);
    printf("number of split bigrams is %d\n",Bigrams);
    int numNames=getNumNames(names);
    printf("number of names is %d\n",numNames);
    // place encoded bigrams into x[] and y[]
    int *X=(int *)calloc(Bigrams*BLOCKSIZE,sizeof(int)); // 182464 x 3
    int *Y=(int *)calloc(Bigrams,sizeof(int)); // 182464 x 1
    encodeX_Y(names,alphabet,X,Y);
    float *emb=(float *)calloc(Bigrams*BLOCKSIZE*DIMENSIONS,sizeof(float)); // Nx3x10
    //embed X into C giving embedded array (emb)
    embedCX(emb,C,X,Bigrams);
    float *Ho = multAB(emb,W1,Bigrams,INNODES,INNODES,HIDNODES);
    addBias(Ho,B1,Bigrams,BIAS1); // add bias vector to each row
    // now attempt batch normalization
    //printE("Ho",Ho,Bigrams,HIDNODES);
    float *Hn = normalizeH(Ho,running_mean,running_var,1,Bigrams);
    // multiply hNormal by the Gain
    multMxV(Hn,bnGain,Bigrams,HIDNODES);
    // now add bnBias to hNormXbnGain
    addBias(Hn,bnBias,Bigrams,HIDNODES);
    // use activation function tanh on each element in the hidden layer
    // this will give us results between -1 and +1
    addTanh(Hn,Bigrams,HIDNODES);
    // now get the logits (log counts)
    float *logits=multAB(Hn,W2,Bigrams,HIDNODES,HIDNODES,OUTNODES); // 32 x 27 array
    // add the bias to each row
    addBias(logits,B2,Bigrams,BIAS2);
    float loss = crossEntropy(logits,Y,Bigrams,OUTNODES);
    free(X);free(Y);free(emb);free(Ho);free(Hn);free(logits);
    return loss;
}