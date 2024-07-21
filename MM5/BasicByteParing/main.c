#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>

#include "helpers.h"
#include "printers.h"
#include "normalize.h"
#include "forepass.h"
#include "backpass.h"
#include "testing.h"
#include "matrix.h"
#include "bytepair.h"
#include "params.h"

int main() {
    setbuf(stdout,0);
    // we'll have to use GNU Scientific Library
    // first select the rng -random number generator
    // used to get random weights and biases
    gsl_rng *pr = gsl_rng_alloc(gsl_rng_mt19937);
    srand(42); // initialize rng - used to get random minibatch
    // start byte pairing
    // get ALL raw bytes from the file
    int numRawBytes=getFileSize(FILENAME);
    int *rawBytes=getRawBytes(FILENAME,numRawBytes);
    printf("Length of input is %d\n",numRawBytes);
    // get the alphabet (unique bytes)
    int lenAlpha=0;
    int *alphabet=getAlphabet(rawBytes,numRawBytes,&lenAlpha);
    printf("initial length of Alphabet is %d\n",lenAlpha);
    printInt("alphabet",alphabet,1,lenAlpha);
    // get the next largest byte value
    int nextValue=getMaxAlpha(alphabet,lenAlpha);
    nextValue++;
    int history[MERGES][4]={{0}};
    // do the merges
    int newlenAlpha,n1=1,n2=1,numPairs=0;
    printf("Merging ");
    for(int i=0;i<MERGES;i++){
        printf(".");
    	newlenAlpha=getBytePair(rawBytes,numRawBytes,alphabet,lenAlpha,&n1,&n2,&numPairs);
    	if(newlenAlpha > lenAlpha){
            lenAlpha=newlenAlpha;
            alphabet=updateAlphabet(alphabet,lenAlpha,nextValue);
			history[i][0]=numPairs;
			history[i][1]=nextValue++;
			history[i][2]=alphabet[n1];
			history[i][3]=alphabet[n2];
			updateLine(rawBytes,numRawBytes,history[i]);
            numRawBytes -= numPairs;
        }else{
            printf("No more duplicate byte pairs!\n");
            break;
        }
    }
    printf("\n");
    printf("lenAlpha after merging is %d\n",lenAlpha);
    printInt("alphabet",alphabet,1,lenAlpha); 
    printf("Number of raw bytes = %d\n",numRawBytes);
    // encode raw data into alphabet index
    encodeRawData(rawBytes,numRawBytes,alphabet,lenAlpha);
    // encode line into X and Y
    int *allX=(int *)calloc((numRawBytes-BLOCKSIZE)*BLOCKSIZE,sizeof(int));
    int *allY=(int *)calloc(numRawBytes-BLOCKSIZE,sizeof(int));
    encodeX_Y(rawBytes,numRawBytes,allX,allY);
    // number of DIMENSIONS for embedding table
    printf("Dimensions is %d\n",DIMENSIONS);
    // how many characters are needed to predict the next one
    printf("Blocksize is %d\n",BLOCKSIZE);
    // let's begin by creating a lookup table of 27 rows and 2 columns
    float *C = makeTable(pr,lenAlpha,DIMENSIONS); 
    printf("Our lookup table size is %d x %d\n", lenAlpha,DIMENSIONS);
    // let's make some weights and biases
    printf("The size of the input  is %d\n",INNODES);
    printf("The size of the hidden is %d\n",HIDNODES);
    printf("The size of the bias 1 is %d\n",BIAS1);
    printf("The size of the output is %d\n",lenAlpha);
    printf("The size of the bias 2 is %d\n",lenAlpha);
    // Create our weights and biases
    float *B1=makeTable(pr,1,BIAS1);
    float *W1=makeTable(pr,BLOCKSIZE*DIMENSIONS,HIDNODES); 
    for(int i=0;i<BLOCKSIZE*DIMENSIONS*HIDNODES;i++)
        W1[i] /= sqrtf(BLOCKSIZE*DIMENSIONS);
    float *W2=makeTable(pr,HIDNODES,lenAlpha);
    for(int i=0;i<HIDNODES*lenAlpha;i++)
        W2[i] = W2[i] * 0.1;
    float *B2=makeTable(pr,1,lenAlpha);
    // adding in gain and bias to further tweak the hidden layer
    // note: we'll need to compute their gradients and update
    // them during back propagation.
    float *bnGain = (float *)calloc(BNGAIN,sizeof(float));
    // let's set them all to 1's
    for(int i=0;i<BNGAIN;i++) bnGain[i]=1;
	// leave this bias at 0's
    float *bnBias=(float *)calloc(BNBIAS,sizeof(float));
    // running mean and var used in makeMore
    float *running_mean = calloc(HIDNODES,sizeof(float));
    float *running_var = calloc(HIDNODES,sizeof(float));
    // set running var to 1's
    for(int i=0;i<HIDNODES;i++)
        running_var[i] = 1.0;
    printf("The number of Parameters is %d\n",BLOCKSIZE*DIMENSIONS*HIDNODES+BIAS1+
                HIDNODES*lenAlpha+lenAlpha+DIMENSIONS*lenAlpha+BNGAIN+BNBIAS);
    printf("The batch size is %d\n",BATCHSIZE);
    // make room for our inputs and outputs
    int *X=(int *)calloc(BATCHSIZE*BLOCKSIZE,sizeof(int)); 
    int *Y=(int *)calloc(BATCHSIZE,sizeof(int));
    // make room for the embedded table
    // some declarations
    float   *H,*logits,*probs,*B2grads,*W2grads,*dLdYp,*dhdz,
            *B1grads,*W2T,*W1T,*dldh,*dldz,*W1grads,*Cgrads,*Egrads,
            *biasGrads,*gainGrads,*dldHn,*mu,*Mu,*var,
            *dbndiff,*M,*temp,*dldmu,*dldHo,*dldHo1,*dldmu1,
            *var_inv,*Hn,*Ho,*dLdS2,*emb;
    float learningRate = INITIAL_LR;
    float loss=0.0;
    printf("The starting learning rate is %5.3lf\n",INITIAL_LR);
    for(int loop=0;loop<200000;loop++){
        if(loop >= 100000) learningRate = -0.01;
        
        ////////////////    start forward pass   ////////////////

        getMiniBatch(numRawBytes-BLOCKSIZE,allX,allY,X,Y);
        //printInt("X",X,BATCHSIZE,BLOCKSIZE);
        //printInt("Y",Y,1,BATCHSIZE);
        //exit(1);
        //embed X into C giving embedded array (emb)
        emb=(float *)calloc(BATCHSIZE*BLOCKSIZE*DIMENSIONS,sizeof(float));
        embedCX(emb,C,X,BATCHSIZE);
        // compute the hidden layer 
        Ho=multAB(emb,W1,BATCHSIZE,INNODES,INNODES,HIDNODES);
        // add bias to hidden layer
        addBias(Ho,B1,BATCHSIZE,BIAS1);
        // now attempt batch normalization
        Hn=normalizeH(Ho,running_mean,running_var,0,BATCHSIZE);
        // copy Hn to H
        H=(float *)calloc(BATCHSIZE*HIDNODES,sizeof(float));
        memcpy(H,Hn,BATCHSIZE*HIDNODES*sizeof(float));
        // multiply hNormal by the Gain
        multMxV(H,bnGain,BATCHSIZE,HIDNODES);
        // now add bnBias to hNormXbnGain
        addBias(H,bnBias,BATCHSIZE,HIDNODES);
        // use activation function tanh 
        addTanh(H,BATCHSIZE,HIDNODES);
        // now get the logits (log counts)
        logits=multAB(H,W2,BATCHSIZE,HIDNODES,HIDNODES,lenAlpha);
        // add bias to the logits
        addBias(logits,B2,BATCHSIZE,lenAlpha);
        // get the loss
        loss=crossEntropy(logits,Y,BATCHSIZE,lenAlpha);
        if(loop%10000 == 0)
            printf("%d\t Loss is %.5lf\n",loop,loss);
        
        ////////////////   start back propagation   ////////////////

        probs=softMax(logits,BATCHSIZE,lenAlpha);
        free(logits);
        dLdYp=getdLdYp(probs,Y,lenAlpha);
        free(probs);
        // let's get the B2 gradients
        B2grads=getB2grads(dLdYp,lenAlpha);
        // let's get the W2 gradients
        W2grads=getW2grads(H,dLdYp,lenAlpha);
        //printE("W2grads",W2grads,200,27);
        dhdz=getdhdz(H);
        free(H);
        // transpose W2
        W2T=transpose(W2,HIDNODES,lenAlpha);
        // get dldh
        dldh=multAB(dLdYp,W2T,BATCHSIZE,lenAlpha,lenAlpha,HIDNODES);
        free(dLdYp);free(W2T);
        // get dldz
        dldz=AxB(dldh,dhdz,BATCHSIZE,HIDNODES);
        free(dhdz);free(dldh);
        // get the bias and gain gradients
        biasGrads=getBiasGrads(dldz);
        gainGrads=getGainGrads(dldz,Hn);
        free(Hn);
        dldHn=getdldHn(dldz,bnGain);
        free(dldz);
        mu=getHmean(Ho);
        Mu=getHmean1(Ho,mu);
        M=AxB(dldHn,Mu,BATCHSIZE,HIDNODES);
        var=getHvar(Ho,mu);
        free(mu);free(Ho);
        temp=(float *)calloc(HIDNODES,sizeof(float));
        for(int i=0;i<HIDNODES;i++)
            temp[i] = -0.5 * powf(var[i]+EPSILON,-1.5);
        dLdS2=multVM(temp,M,BATCHSIZE,HIDNODES);
        free(temp);free(M);
        var_inv=(float *)calloc(HIDNODES,sizeof(float));
        for(int i=0;i<HIDNODES;i++)
            var_inv[i] = -1.0/sqrtf(var[i]+EPSILON);
        dbndiff=VxMsum(var_inv,dldHn,BATCHSIZE,HIDNODES);
        for(int i=0;i<BATCHSIZE*HIDNODES;i++)
            Mu[i] *= -2.0;
        temp=(float *)calloc(HIDNODES,sizeof(float));
        for(int i=0;i<BATCHSIZE;i++)
            for(int j=0;j<HIDNODES;j++)
                temp[j] += *(Mu+i*HIDNODES+j);
        for(int i=0;i<HIDNODES;i++)
            temp[i] /= BATCHSIZE;
        dldmu=multVxV(dLdS2,temp,HIDNODES);
        free(temp);
        addV2V(dldmu,dbndiff,HIDNODES);
        free(dbndiff);
        var_inv=(float *)calloc(HIDNODES,sizeof(float));
        for(int i=0;i<HIDNODES;i++)
            var_inv[i] = 1.0/sqrtf(var[i]+EPSILON);
        dldHo=multMV(dldHn,var_inv,BATCHSIZE,HIDNODES);
        free(var_inv); free(dldHn); free(var);
        for(int i=0;i<BATCHSIZE*HIDNODES;i++)
            Mu[i] /= -(BATCHSIZE-1);
        dldHo1=multVxM(dLdS2,Mu,BATCHSIZE,HIDNODES);
        dldmu1=(float *)calloc(HIDNODES,sizeof(float));
        for(int i=0;i<HIDNODES;i++)
            dldmu1[i] = dldmu[i]/(BATCHSIZE);
        addM2M(dldHo,dldHo1,BATCHSIZE,HIDNODES);
        addV2M(dldmu1,dldHo,BATCHSIZE,HIDNODES);
        free(dldHo1);free(dldmu1);free(Mu),free(dLdS2),free(dldmu);
        B1grads=getB1grads(dldHo);
        W1grads=getW1grads(dldHo,emb);
        free(emb);
        W1T=transpose(W1,INNODES,HIDNODES);
        // get the embedded gradients
        Egrads=multAB(dldHo,W1T,BATCHSIZE,HIDNODES,HIDNODES,INNODES); 
        free(W1T); free(dldHo);
        // get the lookup table C gradients
        Cgrads=getCgrads(Egrads,X,lenAlpha);
        free(Egrads);
        ////////////////   update weights and biases  ////////////////

        update(W2,W2grads,HIDNODES,lenAlpha,learningRate);
        update(B2,B2grads,1,lenAlpha,learningRate);
        update(W1,W1grads,INNODES,HIDNODES,learningRate);
        update(B1,B1grads,1,BIAS1,learningRate);
        update(C, Cgrads, lenAlpha,DIMENSIONS,learningRate);
        update(bnBias,biasGrads,1,BNBIAS,learningRate);
        update(bnGain,gainGrads,1,BNGAIN,learningRate);

        free(Cgrads);
        free(W1grads);
        free(B1grads);
        free(gainGrads);
        free(biasGrads);
        free(B2grads);
        free(W2grads);
    }
    printf("Final loss after training is %7.4f\n",loss);
    printf("The final learning rate is %5.3lf\n",learningRate);
    return 0;
}
