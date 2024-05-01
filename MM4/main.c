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
#include "params.h"

int main() {
    setbuf(stdout,0);
    // we'll have to use GNU Scientific Library
    // first select the rng -random number generator
    // used to get random weights and biases
    gsl_rng *pr = gsl_rng_alloc(gsl_rng_mt19937);
    srand(42); // initialize rng - used to get random minibatch
    char *allNames=getNames("names.txt");
    int numNames=getNumNames(allNames);
    printf("The total number of names is %d\n",numNames);
    int maxBigrams = strlen(allNames) - 1;
    printf("Total number of bigrams is %d\n",maxBigrams);
    // split data into 3 sets - train, develop, test
    // 80% - 10% - 10%
    int first=1,last=ceil(0.8*numNames); // 1 - 25627 = 25627
    char *text4Train = getSetOfNames(allNames,first,last-1);
    printf("BiGrams in Train Set is %ld\n",strlen(text4Train));
    first=last; last=first-1+0.1*numNames; // 25628 - 28830 = 3203
    char *text4Dev   = getSetOfNames(allNames,first,last);
    printf("BiGrams in Dev Set is %ld\n",strlen(text4Dev));
    first=last+1;last=numNames;           // 28831 - 32033 = 3203
    char *text4Test  = getSetOfNames(allNames,first,last);
    printf("BiGrams in Test Set is %ld\n",strlen(text4Test));
    numNames=getNumNames(text4Dev);
    printf("Number of names in the development set = %d\n",numNames);
    numNames=getNumNames(text4Test);
    printf("Number of names in the test set = %d\n",numNames);
    // start with the training set
    numNames=getNumNames(text4Train);
    printf("Number of names in the training set = %d\n",numNames);
    char *alphabet=getAlphabet(text4Train);
    printf("Size of alphabet is %d\n",LENALPHA);
    // number of DIMENSIONS for embedding table
    printf("Dimensions is %d\n",DIMENSIONS);
    // how many characters are needed to predict the next one
    printf("Blocksize is %d\n",BLOCKSIZE);
    // let's begin by creating a lookup table of 27 rows and 2 columns
    float *C = makeTable(pr,LENALPHA,DIMENSIONS); 
    //print("C",C,LENALPHA,DIMENSIONS);
    //exit(1);
    printf("Our lookup table size is %d x %d\n", LENALPHA,DIMENSIONS);
    // let's make some weights and biases
    printf("The size of the input  is %d\n",INNODES);
    printf("The size of the hidden is %d\n",HIDNODES);
    printf("The size of the bias 1 is %d\n",BIAS1);
    printf("The size of the output is %d\n",OUTNODES);
    printf("The size of the bias 2 is %d\n",BIAS2);
    // Create our weights and biases
    float *B1=makeTable(pr,1,BIAS1);
    float *W1=makeTable(pr,BLOCKSIZE*DIMENSIONS,HIDNODES); 
    for(int i=0;i<BLOCKSIZE*DIMENSIONS*HIDNODES;i++)
        W1[i] /= sqrtf(30);
    float *W2=makeTable(pr,HIDNODES,OUTNODES);
    for(int i=0;i<HIDNODES*OUTNODES;i++)
        W2[i] = W2[i] * 0.1;
    //printE("W1",W1,BLOCKSIZE*DIMENSIONS,HIDNODES);
    float *B2=makeTable(pr,1,BIAS2);
    //print("B2",B2,1,27);
    //for(int i=0;i<BIAS2;i++)
    //    B2[i] = 0.0;
    // adding in gain and bias to further tweak the hidden layer
    // note: we'll need to compute their gradients and update
    // them during back propagation.
    float *bnGain = (float *)calloc(BNGAIN,sizeof(float));
    // let's set them all to 1's
    for(int i=0;i<BNGAIN;i++) bnGain[i]=1;
	// leave this bias at 0's
    float *bnBias=(float *)calloc(BNBIAS,sizeof(float));
    // running mean and var used in splitLoss and makeMore
    float *running_mean = calloc(HIDNODES,sizeof(float));
    float *running_var = calloc(HIDNODES,sizeof(float));
    // set running var to 1's
    for(int i=0;i<HIDNODES;i++)
        running_var[i] = 1.0;
    printf("The number of Parameters is %d\n",BLOCKSIZE*DIMENSIONS*HIDNODES+BIAS1+
                HIDNODES*OUTNODES+BIAS2+DIMENSIONS*LENALPHA+BNGAIN+BNBIAS);
    printf("The batch size is %d\n",BATCHSIZE);
    int *allX=(int *)calloc((strlen(allNames))*BLOCKSIZE,sizeof(int));
    int *allY=(int *)calloc((strlen(allNames)),sizeof(int));
    // encode all training bigrams
    encodeX_Y(text4Train,alphabet,allX,allY);
    // make room for our inputs and outputs
    int *X=(int *)calloc(BATCHSIZE*BLOCKSIZE,sizeof(int)); 
    int *Y=(int *)calloc(BATCHSIZE,sizeof(int));
    // make room for our embedded table
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

        getMiniBatch(text4Train,allX,allY,X,Y);
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
        logits=multAB(H,W2,BATCHSIZE,HIDNODES,HIDNODES,OUTNODES);
        // add bias to the logits
        addBias(logits,B2,BATCHSIZE,BIAS2);
        // get the loss
        loss=crossEntropy(logits,Y,BATCHSIZE,OUTNODES);
        if(loop%10000 == 0)
            printf("%d\t Loss is %.5lf\n",loop,loss);
        
        ////////////////   start back propagation   ////////////////

        probs=softMax(logits,BATCHSIZE,LENALPHA);
        dLdYp=getdLdYp(probs,Y);
        // let's get the B2 gradients
        B2grads=getB2grads(dLdYp);
        // let's get the W2 gradients
        W2grads=getW2grads(H,dLdYp);
        //printE("W2grads",W2grads,200,27);
        dhdz=getdhdz(H);
        free(H);
        // transpose W2
        W2T=transpose(W2,HIDNODES,LENALPHA);
        // get dldh
        dldh=multAB(dLdYp,W2T,BATCHSIZE,LENALPHA,LENALPHA,HIDNODES);
        free(dLdYp);free(W2T);
        // get dhdz
        dldz=AxB(dldh,dhdz,BATCHSIZE,HIDNODES);
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
        var_inv=(float *)calloc(HIDNODES,sizeof(float));
        for(int i=0;i<HIDNODES;i++)
            var_inv[i] = 1.0/sqrtf(var[i]+EPSILON);
        dldHo=multMV(dldHn,var_inv,BATCHSIZE,HIDNODES);
        free(var_inv);
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
        W1T=transpose(W1,INNODES,HIDNODES);
        // get the embedded gradients
        Egrads=multAB(dldHo,W1T,BATCHSIZE,HIDNODES,HIDNODES,INNODES); 
        // get the lookup table C gradients
        Cgrads=getCgrads(Egrads,X);

        //printE("dldHo",dldHo,32,200);
        //exit(1);

        ////////////////   update weights and biases  ////////////////

        update(W2,W2grads,HIDNODES,OUTNODES,learningRate);
        update(B2,B2grads,1,BIAS2,learningRate);
        update(W1,W1grads,INNODES,HIDNODES,learningRate);
        update(B1,B1grads,1,BIAS1,learningRate);
        update(C, Cgrads, LENALPHA,DIMENSIONS,learningRate);
        update(bnBias,biasGrads,1,BNBIAS,learningRate);
        update(bnGain,gainGrads,1,BNGAIN,learningRate);

        free(W2grads);free(W1grads);free(B1grads);free(B2grads);
        free(Egrads);free(Cgrads);free(biasGrads);free(gainGrads);
        free(dldHo);
    }
    printf("Final loss after training is %7.4f\n",loss);
    printf("The final learning rate is %5.3lf\n",learningRate);
    loss=splitLoss(text4Train,C,W1,B1,W2,B2,alphabet,
                bnGain,bnBias,running_mean,running_var);
    printf("Loss for Train is %7.4lf\n",loss);
    loss=splitLoss(text4Dev,C,W1,B1,W2,B2,alphabet,
                bnGain,bnBias,running_mean,running_var);
    printf("Loss for Dev   is %7.4lf\n",loss);
    //loss=splitLoss(test4Test,C,W1,B1,W2,B2,alphabet,
    //            bnGain,bnBias,running_mean,running_var);
    //printf("Loss for Test  is %7.4lf\n",loss);
    makeMore(alphabet,C,W1,B1,W2,B2,20,
                bnGain,bnBias,running_mean,running_var);
    return 0;
}
