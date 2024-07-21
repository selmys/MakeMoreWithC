#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "normalize.h"
#include "params.h"

float *getHmean(float *H){
    // compute the mean of each column of H (BATCHSIZE x HIDNODES)
    float *mean=(float *)calloc(HIDNODES,sizeof(float));
    for(int i=0;i<BATCHSIZE;i++)
        for(int j=0;j<HIDNODES;j++)
            mean[j] += *(H+i*HIDNODES+j);
    for(int k=0;k<HIDNODES;k++)
        mean[k] /= BATCHSIZE;
    return mean;
}
float *getHmean1(float *Ho, float *Mu){
    // subtract the mean from each element in the hidden layer 
    float *mean=(float *)calloc(BATCHSIZE*HIDNODES,sizeof(float));
    for(int i=0;i<BATCHSIZE;i++)
        for(int j=0;j<HIDNODES;j++)
            *(mean+i*HIDNODES+j) = *(Ho+i*HIDNODES+j) - Mu[j];
    return mean; // BATCHSIZExHIDNODES
}
float *getHvar(float *H, float *mean){
    // compute the variance - sigma squared = (x - mean) squared
    float *hVar=(float *)calloc(HIDNODES,sizeof(float));
    for(int i=0;i<BATCHSIZE;i++)
        for(int j=0;j<HIDNODES;j++)
            hVar[j] += (*(H+i*HIDNODES+j) - mean[j]) * (*(H+i*HIDNODES+j) - mean[j]);
    for(int k=0;k<HIDNODES;k++)
        hVar[k] /= BATCHSIZE -1; //Bessel's correction 
    return hVar;
}
float *getHdev(float *hVar){
    // get the standard deviation --> sqrt(variance)
    float *hDev=(float *)calloc(HIDNODES,sizeof(float));
    for(int i=0;i<HIDNODES;i++)
        hDev[i] = sqrtf(hVar[i]+EPSILON);
    return hDev;
}
float *getHnormal(float *Ho,float *hMean,float *sDev,int batchSize){
    // normalize Ho giving Hn
    float *Hn=calloc(batchSize*HIDNODES,sizeof(float));
    for(int i=0;i<batchSize;i++)
        for(int j=0;j<HIDNODES;j++)
           *(Hn+i*HIDNODES+j) = (*(Ho+i*HIDNODES+j) - hMean[j])/sDev[j];
    return Hn; // BATCHSIZExHIDNODES
}
float *normalizeH(float *Ho,float *running_mean,float *running_var,int testing,int batchSize){
    float *hMean, *hVar, *sDev, *Hn;
    if(testing){
        hMean = running_mean;
        hVar = running_var;    
        sDev=getHdev(hVar);
        Hn=getHnormal(Ho,hMean,sDev,batchSize);
        free(sDev);
    } else {
        // for each value in Ho we subtract the mean and divide by the standard deviation
        // note: sDev = variance^0.5
        // so Xi = (Xi - mean)/sDev
        // start by getting the average (mean) of every column of H
        hMean=getHmean(Ho);
        // update the running mean
        for(int i=0;i<HIDNODES;i++)
            running_mean[i] = (1.0 - MOMENTUM) * running_mean[i] + MOMENTUM * hMean[i];
        // now get the variance
        hVar=getHvar(Ho,hMean);
        // update the running variance
        for(int i=0;i<HIDNODES;i++)
            running_var[i] = (1.0 - MOMENTUM) * running_var[i] + MOMENTUM * hVar[i];
        // get the standard deviation --> sqrt(variance + EPSILON)
        sDev=getHdev(hVar);
        Hn=getHnormal(Ho,hMean,sDev,batchSize);
        free(hMean); free(hVar); free(sDev); 
    }
    return Hn; 
}