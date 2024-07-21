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

double *probs2d(float *probs,int lenAlpha){
    double *p=(double *)calloc(lenAlpha,sizeof(double));
    for(int i=0;i<lenAlpha;i++)
        p[i]=probs[i];
    return p;
}
int *makeMore(float *C,float *W1,float *B1,float *W2,float *B2,int count,
                float *bnGain,float *bnBias,float *running_mean,
                float *running_var,int *alphabet,int lenAlpha,int numRawBytes){
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
    unsigned int *n = (unsigned int *)calloc(lenAlpha,sizeof(int));
    float *emb=(float *)calloc(INNODES,sizeof(float));
    int *out=(int *)calloc(numRawBytes,sizeof(int));
    int p=0;
    int numBigrams=1;
    int X[BLOCKSIZE];
    for(int a=0;a<BLOCKSIZE;a++) X[a]=0;
    for(int i=0;i<count;i++){
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
        logits=multAB(Hn,W2,numBigrams,HIDNODES,HIDNODES,lenAlpha);
        // add bias to the logits
        addBias(logits,B2,numBigrams,lenAlpha);
        // get the probabilities
        probs=softMax(logits,numBigrams,lenAlpha);
        dprobs=probs2d(probs,lenAlpha);
        gsl_ran_multinomial(pr,lenAlpha,1,dprobs,n);
        // find which element of n has a 1
        for(int j=0;j<lenAlpha;j++){
            if(n[j]!=0) ix=j;
        }
        //printf("%d ",alphabet[ix]);
        out[p++]=ix;
        for(k=0;k<BLOCKSIZE-1;k++)
            X[k]=X[k+1];
        X[k]=ix;
    }
    printf("\n");
    return out;
}