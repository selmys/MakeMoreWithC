#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "printers.h"
#include "backpass.h"
#include "matrix.h"
#include "params.h"

float *getdLdYp(float *probs,int *Y,int lenAlpha){
    // compute the gradient of loss w.r.t predicted output
    // update probabilities to derivatives
    // probs[i][j] = probs[i][j] - 1.0 only for Y[i] == j
    float *dLdYp=(float *)calloc(BATCHSIZE*lenAlpha,sizeof(float));
    memcpy(dLdYp,probs,BATCHSIZE*lenAlpha*sizeof(float));
    for(int i=0;i<BATCHSIZE;i++)
        for(int j=0;j<lenAlpha;j++)
            if(Y[i] == j)
                *(dLdYp+i*lenAlpha+j) = *(dLdYp+i*lenAlpha+j) - 1.0;
    return dLdYp;
}
float *getB2grads(float *dLdYp,int lenAlpha){
    float *B2grads = sumCols(dLdYp,BATCHSIZE,lenAlpha);
    // divide B2grads by BATCHSIZE
    for(int i=0;i<lenAlpha;i++)
        B2grads[i] /= BATCHSIZE;
    return B2grads;
}
float *getW2grads(float *Ha,float *dLdYp,int lenAlpha){
    // transpose the Ha array so we can multiply
    float *HT = transpose(Ha,BATCHSIZE,HIDNODES);
    float *W2grads = multAB(HT,dLdYp,HIDNODES,BATCHSIZE,BATCHSIZE,lenAlpha);
    free(HT);
    // divide W2grads by BATCHSIZE
    for(int i=0;i<HIDNODES;i++)
        for(int j=0;j<lenAlpha;j++)
            *(W2grads+i*lenAlpha+j) /= BATCHSIZE;
    return W2grads;
}
float *getBiasGrads(float *dldz){
    /*float *biasGrads=(float *)calloc(BNBIAS,sizeof(float));
    // add up the columns of dldz
    for(int i=0;i<BATCHSIZE;i++)
        for(int j=0;j<HIDNODES;j++)
            biasGrads[j] += *(dldz+i*HIDNODES+j);
    */
    float *biasGrads = sumCols(dldz,BATCHSIZE,HIDNODES);
    for(int k=0;k<HIDNODES;k++)
        biasGrads[k] /= BATCHSIZE;
    return biasGrads; // 1xBNBIAS
}
float *getGainGrads(float *dldz,float *H){
    float *gainGrads=AxBsum(dldz,H,BATCHSIZE,HIDNODES);
    // divide gainGrads by BATCHSIZE
    for(int i=0;i<BNGAIN;i++)
        gainGrads[i] /= BATCHSIZE;
    return gainGrads;
}
float *getdhdz(float *H){
    // get dh/dz ---> derivative of tanh is 1 - tanh^2
    float *dhdz=(float *)calloc(BATCHSIZE*HIDNODES,sizeof(float));  // 1-H^2
    for(int i=0;i<BATCHSIZE;i++)
        for(int j=0;j<HIDNODES;j++)
            *(dhdz+i*HIDNODES+j) = 1.0 - ((*(H+i*HIDNODES+j)) * (*(H+i*HIDNODES+j)));
    return dhdz;
}
float *getdldHn(float *dldz,float *bnGain){
    float *dLdHn=multMV(dldz,bnGain,BATCHSIZE,HIDNODES); 
    // divide dLdHn by numBigrams
    for(int i=0; i<BATCHSIZE*HIDNODES;i++)
        dLdHn[i] /= BATCHSIZE;
    return dLdHn;
}
float *getB1grads(float *dldHo){
    // make room for B1grads
    /*float *B1grads = (float *)calloc(BIAS1,sizeof(float));
    // add all columns of dldHo into B1grads
    for(int i=0;i<BATCHSIZE;i++)
        for(int j=0;j<BIAS1;j++)
            B1grads[j] += *(dldHo+i*BIAS1+j);
    */
    float *B1grads = sumCols(dldHo,BATCHSIZE,HIDNODES);
    // divide B1grads by 32
    //for(int i=0;i<BIAS1;i++)
      //  B1grads[i] /= BATCHSIZE;
    return B1grads;
}
float *getW1grads(float *dldh,float *emb){
    float *embT = transpose(emb,BATCHSIZE,INNODES);
    float *W1grads = multAB(embT,dldh,INNODES,BATCHSIZE,BATCHSIZE,HIDNODES);
    // finally divide gradients by 32
    for(int i=0;i<INNODES;i++)
        for(int j=0;j<HIDNODES;j++)
            *(W1grads+i*HIDNODES+j) /= BATCHSIZE;
    free(embT); 
    return W1grads;
}
float *getCgrads(float *embGrads,int *X,int lenAlpha){
    // basically unembedding the embedded gradients back into the C gradients
    float *Cgrads=(float *)calloc(lenAlpha*DIMENSIONS,sizeof(float)); 
    int Ccol,Ecol;
    for(int i=0;i<BATCHSIZE;i++){
        Ecol=0;
        for(int j=0;j<BLOCKSIZE;j++){
            Ccol = *(X+i*BLOCKSIZE+j);
            for(int k=0;k<DIMENSIONS;k++){
                *(Cgrads+Ccol*DIMENSIONS+k) += *(embGrads+i*INNODES+Ecol++);
            }
        }
    }
    return Cgrads;
}
void update(float *A, float *B, int rows, int cols, float learningRate){
	// update any table
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            *(A+i*cols+j) += learningRate * *(B+i*cols+j);
    return;
}
