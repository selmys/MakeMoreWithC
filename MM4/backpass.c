#include <stdlib.h>
#include <string.h>

#include "backpass.h"
#include "matrix.h"
#include "params.h"

float *getdLdY(float *Ha,float *probs,int *Y,float *W2){
    // 1/BATCHSIZE * sum (Yp -Y)
    float *dLdYp=getdLdYp(probs,Y);
    // transpose W2
    float *W2T = transpose(W2,HIDNODES,OUTNODES); 
    // multiply dLossdYp by W2T
    float *temp=multAB(dLdYp,W2T,BATCHSIZE,LENALPHA,LENALPHA,HIDNODES); 
    // get H ---> derivative of tanh is 1 - tanh^2
    float *H=getdhdz(Ha); 
    // element-wise multiplication
    float *dLdY=AxB(temp,H,BATCHSIZE,HIDNODES); 
    free(dLdYp);free(W2T);free(temp);free(H);
    return dLdY;
}

float *getdLdYp(float *probs,int *Y){
    // compute the gradient of loss w.r.t predicted output
    // update probabilities to derivatives
    // probs[i][j] = probs[i][j] - 1.0 only for Y[i] == j
    float *dLdYp=(float *)calloc(BATCHSIZE*LENALPHA,sizeof(float));
    memcpy(dLdYp,probs,BATCHSIZE*LENALPHA*sizeof(float));
    for(int i=0;i<BATCHSIZE;i++)
        for(int j=0;j<LENALPHA;j++)
            if(Y[i] == j)
                *(dLdYp+i*LENALPHA+j) = *(dLdYp+i*LENALPHA+j) - 1.0;
    return dLdYp;
}
float *getB2grads(float *dLdYp){
    /*// allocate space for B2 gradients
    float *B2grads=(float *)calloc(BIAS2,sizeof(float));
    // add up all the gradients - essentially dL/dYp * dYp/dB2
    for(int i=0;i<BATCHSIZE;i++)
        for(int j=0;j<LENALPHA;j++)
            B2grads[j] += *(dLdYp+i*BIAS2+j);
    */
    float *B2grads = sumCols(dLdYp,BATCHSIZE,LENALPHA);
    // divide B2grads by BATCHSIZE
    for(int i=0;i<LENALPHA;i++)
        B2grads[i] /= BATCHSIZE;
    return B2grads;
}
float *getW2grads(float *Ha,float *dLdYp){
    // transpose the Ha array so we can multiply
    float *HT = transpose(Ha,BATCHSIZE,HIDNODES);
    float *W2grads = multAB(HT,dLdYp,HIDNODES,BATCHSIZE,BATCHSIZE,LENALPHA);
    free(HT);
    // divide W2grads by BATCHSIZE
    for(int i=0;i<HIDNODES;i++)
        for(int j=0;j<LENALPHA;j++)
            *(W2grads+i*LENALPHA+j) /= BATCHSIZE;
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
    //free(embT);   ???
    return W1grads;
}
float *getCgrads(float *embGrads,int *X){
    // basically unembedding the embedded gradients back into the C gradients
    float *Cgrads=(float *)calloc(LENALPHA*DIMENSIONS,sizeof(float)); // 27x2
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
