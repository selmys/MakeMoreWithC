#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>
#include <limits.h>

char * getNames(char *fileName){
    // get names from file and place into a string
    struct stat st;
    int fd = open(fileName,O_RDONLY);
    stat(fileName, &st);
    off_t size = st.st_size;
    // need room for \n at front and \0 at end
    char *names=(char *)calloc(size+2,sizeof(char));
    names[0]='\n';
    int num=read(fd,names+1,size);
    if(num < 0) {
        printf("Cannot read names file!\n");
        exit(2);
    }
    close(fd);
    // change new line characters to periods
    int len = strlen(names);
    for(int i=0; i<len; i++)
        if(names[i] == '\n')names[i] = '.';
    return names;
}
double *makeTable(int size){
    // we'll have to use GNU Scientific Library
    // first select the rng -random number generator- to use
    gsl_rng *pr = gsl_rng_alloc(gsl_rng_mt19937);
    double *C=malloc(size*sizeof(double));
    for(int i=0;i<size;i++)
        *(C+i) = gsl_ran_gaussian(pr,1.0); // standard deviation of 1.0
    return C;
}
char *getAlphabet(char *text){
    // get the list of unique characters from text string
    // assume ASCII so need space for max of 128 characters
    char chars[128] = {0};
    char *letters = (char *)calloc(128,1);
    int len = strlen(text);
    for(int i=0;i<len;i++)
        chars[(int)text[i]] = text[i];
    for(int i=0,j=0;i<128;i++)
        // append characters to letters string
        if(chars[i] != 0)
            letters[j++] = chars[i];
    return letters;
}
int getNumNames(char *names){
    int len=strlen(names);
    int count=0;
    for(int i=1;i<len;i++)
        if(names[i] == '.') count++;
    return count;
}
char *getOneName(int n,char *names){
    // return one name without the periods
    // n==3 will return the 3rd name
    int i=1;
    char *temp=(char *)malloc(strlen(names)+1);
    strcpy(temp,names);
    char *oneName=strtok(temp+1,".");
    while(oneName != NULL){
        if(i >= n) break;
        oneName=strtok(NULL,".");
        i++;
    }
    return oneName;
}
char *getSomeNames(char *text, int howMany){
    // words are deliminated by periods
    // return a string of words starting from the
    // beginning of the text string
    int to=0,count=0;
    for(int i=1;i<strlen(text);i++){
        if(text[i] == '.'){
            count++;
            if(count >= howMany){
                to = i;
                break;
            }
        }
    }
    char *names = (char *)malloc(++to);
    strncpy(names,text,to);
    return names;
}
char *getSetOfNames(char *text, int fm, int to){
    // return a string of names from (fm) to (to)
    int start=0,stop=0,count=0;
    for(int i=0;i<strlen(text);i++){
        if(text[i] == '.'){
            count += 1;
            if(count == fm){
                start = i;
            }else{
                if(count-1 == to)
                    stop = i;
            }
        }
    }
    char *names = calloc(stop-start+2,1);
    strncpy(names,text+start,stop-start+1);
    return names;
}
int getIndex(char *s, char c){
    // returns the index of c in s
    for(int i=0;i<strlen(s);i++)
        if(s[i] == c) return i;
    return -1;
}
void printBigrams(int blocksize, char *names){
    int numNames=getNumNames(names);
    int k;
    char *s=(char *)malloc(30); // max length of any name assumed to be 30
    for(int i=1;i<numNames+1;i++){
        char *w=getOneName(i,names);
        printf("%s\n",w);
        *s=0;
        strcat(s,"...");
        strcat(s,w); 
        strcat(s,".");
        for(int j=0;j<strlen(w)+1;j++){
            for(k=0;k<blocksize;k++)
                printf("%c",s[k]);
            printf(" ---> %c\n",s[k]);
            s=s+1;
        }
    }
}
void printEmb(double *emb, int l, int w, int h){
    printf("=========== emb ============\n");
    for(int i=0;i<l;i++){
        for(int j=0;j<w;j++){
            for(int k=0;k<h;k++){
                printf("%7.4lf ",*(emb+i*w*h+j*h+k));
            }
            printf("\n");
        }
        printf("\n");
    }
    return;
}
double *formatEmb(double *emb, int l, int w, int h){
    // reformat the embedded matrix
    double *R=calloc(l*w*h,sizeof(double));
    int p=0;
    for(int i=0;i<l;i++){
        for(int j=0;j<w;j++){
            for(int k=0;k<h;k++){
                R[p] = *(emb+i*w*h+j*h+k);
                p++;
            }
        }
    }
    return R;
}
void printEncodedBigrams(int numBigrams,int blocksize,int *X,int *Y){
    for(int i=0;i<numBigrams;i++){
        for(int j=0;j<blocksize;j++)
            printf("%3d",*(X+i*blocksize+j));
        printf(" ---> %d\n",*(Y+i));
    }
}
void encodeX_Y(char *list,char *alphabet,int *X,int *Y,int blocksize){
    // list is a string of period delimited names
    char *aName; 
    int lenName;
    int ix;
    int m;
    int pX=0,pY=0;
    int context[blocksize];
    int numNames=getNumNames(list);
    for(int i=1;i<=numNames;i++){
        for(int j=0;j<blocksize;j++)
            context[j]=0;
        aName=getOneName(i,list);
        lenName=strlen(aName);
        for(m=0;m<lenName;m++){
            for(int k=0;k<blocksize;k++)
                X[pX++]=context[k];
            ix=getIndex(alphabet,aName[m]);
            Y[pY++]=ix;
            for(int l=0;l<blocksize-1;l++)
                context[l]=context[l+1];
            context[blocksize-1]=ix;
        }
        Y[pY++]=0;
        for(int k=0;k<blocksize;k++)
            X[pX++]=context[k];
    }
    return;
}
void printInt(char *title, int *matrix, int rows, int cols){
    printf("========== %s ===========\n",title);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++)
            printf("%3d ",*(matrix+i*cols+j));
        printf("\n");
    }
    return;
}
void print(char *title, double *matrix, int rows, int cols){
    printf("========== %s ===========\n",title);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++)
            printf(" %.10lf",*(matrix+i*cols+j));
        printf("\n");
    }
    return;
}
void printE(char *title, double *matrix, int rows, int cols){
    printf("========== %s ===========\n",title);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++)
            printf(" %.10e",*(matrix+i*cols+j));
        printf("\n");
    }
    return;
}
void embedCX(double *emb,double *c,int *x,int numBigrams,int blocksize,int dimensions){
    // homebrew embed function
    // step through all values in X (i = numBigrams X blocksize)
    // take the vector from the lookup table (c)
    // and place it into the embedding table (emb)
    int k=0;
    for(int i=0;i<numBigrams*blocksize;i++)
        for(int j=0;j<dimensions;j++)
            emb[k++] = c[x[i]*dimensions+j];
    return;
}
double *AxB(double *A,double *B,int rows,int cols){
    // element-wise multiplication of C = A x B
    double *C = calloc(rows*cols,sizeof(double));
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            *(C+i*cols+j) = *(A+i*cols+j) * *(B+i*cols+j);
    return C;
}
double *multAB(double *A, double *B, int ai, int aj, int bi, int bj) {
    // matrix multiplication
    double sum;
    double *C = (double *)malloc(ai*bj*8);
    for(int i=0;i<ai;i++)
        for(int j=0;j<bj;j++){
            sum = 0.0;
            for(int k=0;k<aj;k++){
                sum += *(A+i*bi+k) * *(B+k*bj+j);
            }
            *(C+i*bj+j) = sum;
        }
    return C;
}
double *transpose(double *a,int r, int c){
    // transpose a matrix
    double *t = calloc(r*c,sizeof(double));
    for (int i=0;i<r;i++) {
		for (int j=0;j<c;j++)
			*(t+j*r+i) = *(a+i*c+j);
	}
    return t;
}
void addBias(double *A,double *B,int rows,int cols){
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            *(A+i*cols+j) += *(B+j);
    return;
}
double *getCounts(double *logits, int rows, int cols){
    // find the largest log count and subtract it from all logits
    //    to prevent exp() overflow with a large log count
    double largest=logits[0];
    int len=rows*cols;
    for(int i=1;i<len;i++)
        if(logits[i] > largest) 
            largest=logits[i];
    // now subtract the largest from the logits
    for(int i=0;i<len;i++)
        logits[i] = logits[i] - largest;
    // change log counts (logits) to counts
    double *counts = (double *)calloc(rows*cols,sizeof(double));
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            *(counts+i*cols+j) = exp(*(logits+i*cols+j));
    return counts;
}
double *getProbs(double *counts, int rows, int cols){
    // change counts to probabilities
    double *probs=(double *)calloc(rows*cols,sizeof(double));
    double sum;
    for(int i=0;i<rows;i++){
        sum=0.0;
        for(int j=0;j<cols;j++)
            sum += *(counts+i*cols+j);
        for(int j=0;j<cols;j++)
            *(probs+i*cols+j) = *(counts+i*cols+j)/sum;
    }
    return probs;
}
void getMiniBatch(char *allNames,int *allX,int *allY,int *X,int *Y,int blocksize,int numBigrams){
    int p,q;
    for(int i=0;i<numBigrams;i++){    
        p=rand()%(strlen(allNames)-1);  
        for(q=0;q<blocksize;q++)
            *(X+i*blocksize+q) = *(allX+p*blocksize+q);
        Y[i] = allY[p];
    }
    return;
}
double *softMax(double *logits, int rows, int cols){
    // get the counts
    double *counts=getCounts(logits,rows,cols);
    // get the probabilities
    double *probs=getProbs(counts,rows,cols);
    free(counts);
    return probs;
}
double *getActuals(double *probs, int *Y, int rows, int cols){
    double *probArray=(double *)malloc(rows*sizeof(double));
    for(int i=0;i<rows;i++)
        probArray[i] = *(probs+i*cols+Y[i]);
    return probArray;
}
double crossEntropy(double *logits, int *Y, int rows, int cols){
    // get the probabilities
    double *probs=softMax(logits,rows,cols);
    // get actual results from the nn
    double *A=getActuals(probs,Y,rows,cols);
    // now get the loss
    // add up the negative log likelihood of the result probabilities
    double nll=0.0;
    for(int i=0;i<rows;i++)
        nll += -log(A[i]);
    // so the average nll is the loss
    double loss = nll/rows;
    free(probs);free(A);
    return loss;
}
void addTanh(double *H,int rows,int cols){
    // tanh is our sigmoid function
    for(int i=0;i<rows;i++)
       for(int j=0;j<cols;j++)
           *(H+i*cols+j) = tanh(*(H+i*cols+j));
    return;
}
double *getdLdYp(double *probs,int *Y,int numBigrams,int lenAlpha){
    // compute the gradient of loss w.r.t predicted output
    // update probabilities to derivatives
    // probs[i][j] = probs[i][j] - 1.0 only for Y[i] == j
    // numBigramsx27
    double *dLdYp=calloc(numBigrams*lenAlpha,sizeof(double));
    memcpy(dLdYp,probs,numBigrams*lenAlpha*8);
    for(int i=0;i<numBigrams;i++)
        for(int j=0;j<lenAlpha;j++)
            if(Y[i] == j)
                *(dLdYp+i*lenAlpha+j) = *(dLdYp+i*lenAlpha+j) - 1.0;
    return dLdYp;
}
double *getB2grads(double *dLdYp,int sizeB2,int numBigrams,int lenAlpha){
    // allocate space for B2 gradients
    double *B2grads=calloc(sizeB2,sizeof(double));
    // add up all the gradients - essentially dL/dYp * dYp/dB2
    for(int i=0;i<lenAlpha;i++)
        for(int j=0;j<numBigrams;j++)
            B2grads[i] += *(dLdYp+j*sizeB2+i);
    // divide B2grads by numBigrams
    for(int i=0;i<lenAlpha;i++)
        B2grads[i] /= numBigrams;
    return B2grads;
}
double *getW2grads(double *H,double *dLdYp,int numBigrams,int hidNodes,int lenAlpha){
    // transpose the H array so we can multiply
    double *HT = transpose(H,numBigrams,hidNodes);
    double *W2grads = multAB(HT,dLdYp,hidNodes,numBigrams,numBigrams,lenAlpha);
    free(HT);
    // divide W2grads by numBigrams
    for(int i=0;i<hidNodes;i++)
        for(int j=0;j<lenAlpha;j++)
            *(W2grads+i*lenAlpha+j) /= numBigrams;
    return W2grads;
}
double *getdhdz(double *H,int numBigrams,int hidNodes){
    // get dh/dz ---> derivative of tanh is 1 - tanh^2
    double *dhdz=calloc(numBigrams*hidNodes,8);  // 1-H^2
    for(int i=0;i<numBigrams;i++)
        for(int j=0;j<hidNodes;j++)
            *(dhdz+i*hidNodes+j) = 1.0 - ((*(H+i*hidNodes+j)) * (*(H+i*hidNodes+j)));
    return dhdz;
}
double *getB1grads(double *dldz,int sizeB1,int numBigrams){
    // make room for B1grads
    double *B1grads = calloc(sizeB1,sizeof(double));
    // add all columns of dldz into B1grads
    for(int i=0;i<numBigrams;i++)
        for(int j=0;j<sizeB1;j++)
            B1grads[j] += *(dldz+i*sizeB1+j);
    // divide B1grads by 32
    for(int i=0;i<sizeB1;i++)
        B1grads[i] /= numBigrams;
    return B1grads;
}
double *getW1grads(double *dldh,double *emb,int numBigrams,int inNodes,int hidNodes){
    double *embT = transpose(emb,numBigrams,inNodes);
    double *W1grads = multAB(embT,dldh,inNodes,numBigrams,numBigrams,hidNodes);
    // finally divide gradients by 32
    for(int i=0;i<inNodes;i++)
        for(int j=0;j<hidNodes;j++)
            *(W1grads+i*hidNodes+j) /= numBigrams;
    //free(embT);   ???
    return W1grads;
}
double *getCgrads(double *embGrads,int *X,int numBigrams,int inNodes,int lenAlpha,int dimensions,int blocksize){
    // basically unembedding the embedded gradients back into the C gradients
    double *Cgrads=calloc(lenAlpha*dimensions,sizeof(double)); // 27x2
    int Ccol,Ecol;
    for(int i=0;i<numBigrams;i++){
        Ecol=0;
        for(int j=0;j<blocksize;j++){
            Ccol = *(X+i*blocksize+j);
            for(int k=0;k<dimensions;k++){
                *(Cgrads+Ccol*dimensions+k) += *(embGrads+i*inNodes+Ecol++);
            }
        }
    }
    // average gradients by dividing by 32
    for(int i=0;i<lenAlpha;i++)
        for(int j=0;j<dimensions;j++)
            *(Cgrads+i*dimensions+j) /= numBigrams;
    return Cgrads;
}
void update(double *A, double *B, int rows, int cols, double learningRate){
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            *(A+i*cols+j) += learningRate * *(B+i*cols+j);
    return;
}
void makeMore(double *C,double *W1,double *B1,double *W2,double *B2,
                int count,char *alphabet,int blocksize,int dimensions,int inNodes,
                int hidNodes,int outNodes,int lenBias1,int lenBias2){
    // let's make count more names
    // but first we need to set up the RNG
    // using GSL (GNU Scientific Library)
    // first select the rng -random number generator- to use
    gsl_rng *pr = gsl_rng_alloc(gsl_rng_ran0);
    // then set the seed to the rng
    gsl_rng_set(pr, 2147483647); // use same as in Andrej's videos
    //gsl_rng_set(pr, time(0));
    int ix,k;
    int lenAlpha=strlen(alphabet);
    double *H,*logits,*probs;
    unsigned int *n = malloc(lenAlpha*sizeof(int));
    double *emb=calloc(inNodes,sizeof(double));
    int numBigrams=1;
    int X[blocksize];
    for(int i=0;i<count;i++){
        for(int a=0;a<blocksize;a++) X[a]=0;
        while(1){
            //embed X into C giving embedded array (emb)
            embedCX(emb,C,X,numBigrams,blocksize,dimensions);
            // compute the hidden neurons
            H=multAB(emb,W1,numBigrams,inNodes,inNodes,hidNodes);
            // add bias to hidden neurons
            addBias(H,B1,numBigrams,lenBias1);
            // use activation function tanh on each element in the hidden layer
            addTanh(H,numBigrams,hidNodes);
            // now get the logits (log counts)
            logits=multAB(H,W2,numBigrams,hidNodes,hidNodes,outNodes);
            // add bias to the logits
            addBias(logits,B2,numBigrams,lenBias2);
            // get the probabilities
            probs=softMax(logits,numBigrams,lenAlpha);
            gsl_ran_multinomial(pr,lenAlpha,1,probs,n);
            // find which element of n has a 1
            for(int j=0;j<lenAlpha;j++){
                if(n[j]!=0) ix=j;
            }
            if(ix != 0)
                printf("%c",alphabet[ix]);
            else
                break;
            for(k=0;k<blocksize;k++)
                X[k]=X[k+1];
            X[k]=ix;
        }
        printf(".\n");
    }
    return;
}
double splitLoss(char *names,double *C,double *W1,double *B1,double *W2,double *B2,char *alphabet,
    // this function does not update any weights or biases or the lookup table
                    int blocksize,int dimensions,int inNodes,int hidNodes,int outNodes,
                    int sizeB1,int sizeB2){
    int Bigrams = strlen(names) - 1;
    printf("number of split bigrams is %d\n",Bigrams);
    int numNames=getNumNames(names);
    printf("number of names is %d\n",numNames);
    // place encoded bigrams into x[] and y[]
    int *X=calloc(Bigrams*blocksize,sizeof(int)); // 282146 x 3
    int *Y=calloc(Bigrams,sizeof(int)); // 282146 x 1
    encodeX_Y(names,alphabet,X,Y,blocksize);
    double *emb=calloc(Bigrams*blocksize*dimensions,sizeof(double)); // 32 x 3 x 2
    //embed X into C giving embedded array (emb)
    embedCX(emb,C,X,Bigrams,blocksize,dimensions);
    double *H = multAB(emb,W1,Bigrams,inNodes,inNodes,hidNodes);
    addBias(H,B1,Bigrams,sizeB1); // add bias vector to each row
    // use activation function tanh on each element in the hidden layer
    // this will give us results between -1 and +1
    addTanh(H,Bigrams,hidNodes);
    // now get the logits (log counts)
    double *logits=multAB(H,W2,Bigrams,hidNodes,hidNodes,outNodes); // 32 x 27 array
    // add the bias to each row
    addBias(logits,B2,Bigrams,sizeB2);
    double loss = crossEntropy(logits,Y,Bigrams,outNodes);
    free(X);free(Y);free(emb);free(H);free(logits);
    return loss;
}
int main() {
    // initialize our rng's
    srand(42); // used in getMiniBatch()
    //gsl_rng *pr = gsl_rng_alloc(gsl_rng_mt19937);
    //gsl_rng_set(pr, 2147483647); // use same as in Andrej's videos
    //gsl_rng_set(pr, time(0));
    char *allNames=getNames("names.txt");
    int numNames=getNumNames(allNames);
    printf("The total number of names is %d\n",numNames);
    int maxBigrams = strlen(allNames) - 1;
    printf("Total number of bigrams is %d\n",maxBigrams);
    // split data into 3 sets - train, develop, test
    // 80% - 10% - 10%
    int first=1,last=ceil(0.8*numNames); // 1 - 25627 = 25627
    char *text4Train = getSetOfNames(allNames,first,last-1);
    first=last; last=first-1+0.1*numNames; // 25628 - 28830 = 3203
    char *text4Dev   = getSetOfNames(allNames,first,last);
    first=last+1;last=numNames;           // 28831 - 32033 = 3203
    char *text4Test  = getSetOfNames(allNames,first,last);
    numNames=getNumNames(text4Dev);
    printf("Number of names in the development set = %d\n",numNames);
    numNames=getNumNames(text4Test);
    printf("Number of names in the test set = %d\n",numNames);
    // start with the training set
    numNames=getNumNames(text4Train);
    printf("Number of names in the training set = %d\n",numNames);
    char *alphabet=getAlphabet(text4Train);
    int lenAlpha=strlen(alphabet);
    printf("Size of alphabet is %d\n",lenAlpha);
    // number of dimensions for embedding table
    int dimensions=10;
    printf("Dimensions is %d\n",dimensions);
    // how many characters are needed to predict the next one
    int blocksize=3;
    printf("Blocksize is %d\n",blocksize);
    // let's begin by creating a lookup table of 27 rows and 2 columns
    double *C = makeTable(lenAlpha*dimensions); 
    //print("C",C,lenAlpha,dimensions);
    //exit(1);
    printf("Our lookup table size is %d x %d\n", lenAlpha,dimensions);
    // let's make some weights and biases
    int hidNodes=200; // number of hidden nodes
    int lenBias1=200; // length of bias 1
    int outNodes=27;  // length of output nodes
    int lenBias2=27;  // length of bias 2
    int inNodes=blocksize*dimensions;
    printf("The size of the input  is %d\n",inNodes);
    printf("The size of the hidden is %d\n",hidNodes);
    printf("The size of the bias 1 is %d\n",lenBias1);
    printf("The size of the output is %d\n",outNodes);
    printf("The size of the bias 2 is %d\n",lenBias2);
    // Create our weights and biases
    double *W1=makeTable(blocksize*dimensions*hidNodes); 
    double *B1=makeTable(lenBias1);
    double *W2=makeTable(hidNodes*outNodes);
    double *B2=makeTable(lenBias2);
    printf("The number of Parameters is %d\n",blocksize*dimensions*hidNodes+lenBias1+
                hidNodes*outNodes+lenBias2+dimensions*lenAlpha);
    // set a minibatch of size=32
    int batchsize = 32;
    printf("The batch size is %d\n",batchsize);
    int *allX=calloc((strlen(allNames))*blocksize,sizeof(int));
    int *allY=calloc((strlen(allNames)),sizeof(int));
    // encode all training bigrams
    encodeX_Y(text4Train,alphabet,allX,allY,blocksize);
    // make room for our inputs and outputs
    int *X=(int *)calloc(batchsize*blocksize,sizeof(int)); 
    int *Y=(int *)calloc(batchsize,sizeof(int));
    // make room for our embedded table
    double *emb=(double *)calloc(batchsize*blocksize*dimensions,sizeof(double));
    // some declarations
    double *H,*logits,*probs,*B2grads,*W2grads,*dLdYp,*dhdz,
            *B1grads,*W2T,*W1T,*dldh,*dldz,*W1grads,*Cgrads,*Egrads;
    double learningRate = -0.1;
    double loss;
    printf("The starting learning rate is %5.3lf\n",learningRate);
    for(int loop=0;loop<200000;loop++){
        if(loop >= 100000) learningRate = -0.01;
        
        ////////////////    start forward pass   ////////////////

        getMiniBatch(text4Train,allX,allY,X,Y,blocksize,batchsize);
        //embed X into C giving embedded array (emb)
        embedCX(emb,C,X,batchsize,blocksize,dimensions);
        // compute thee hidden neurons
        H=multAB(emb,W1,batchsize,inNodes,inNodes,hidNodes);
        // add bias to hidden neurons
        addBias(H,B1,batchsize,lenBias1);
        // use activation function tanh on each element in the hidden layer
        addTanh(H,batchsize,hidNodes);
        // now get the logits (log counts)
        logits=multAB(H,W2,batchsize,hidNodes,hidNodes,outNodes);
        // add bias to the logits
        addBias(logits,B2,batchsize,lenBias2);
        // get the loss
        loss=crossEntropy(logits,Y,batchsize,outNodes);
        if(loop%1000 == 0)
            printf("Loss is %8.5lf\n",loss);
        ////////////////   start back propagation   ////////////////

        probs=softMax(logits,batchsize,lenAlpha);
        // let's get the W2 gradients
        dLdYp=getdLdYp(probs,Y,batchsize,lenAlpha);
        B2grads=getB2grads(dLdYp,lenBias2,batchsize,lenAlpha);
        // printE("B2 Gradients",B2grads,1,lenBias2);
        W2grads=getW2grads(H,dLdYp,batchsize,hidNodes,lenAlpha);
        // printE("W2 Gradients",W2grads,hidNodes,lenAlpha);
        dhdz=getdhdz(H,batchsize,hidNodes);
        // transpose W2
        W2T=transpose(W2,hidNodes,lenAlpha);
        // get dldh
        dldh=multAB(dLdYp,W2T,batchsize,lenAlpha,lenAlpha,hidNodes); 
        // get dhdz
        dldz=AxB(dldh,dhdz,batchsize,hidNodes); // element-wise matrix multiplication
        B1grads=getB1grads(dldz,lenBias1,batchsize);
        // printE("B1 Gradients",B1grads,1,lenBias1);
        W1grads=getW1grads(dldz,emb,batchsize,inNodes,hidNodes);
        // printE("W1 Gradients",W1grads,inNodes,hidNodes);
        W1T=transpose(W1,inNodes,hidNodes);
        // get the embedded gradients
        Egrads=multAB(dldz,W1T,batchsize,hidNodes,hidNodes,inNodes); 
        // get the lookup table C gradients
        Cgrads=getCgrads(Egrads,X,batchsize,inNodes,lenAlpha,dimensions,blocksize);
        // printE("C Gradients",Cgrads,lenAlpha,dimensions);

        ////////////////   update weights and biases  ////////////////

        update(W2,W2grads,hidNodes,outNodes,learningRate);
        update(B2,B2grads,1,lenBias2,learningRate);
        update(W1,W1grads,inNodes,hidNodes,learningRate);
        update(B1,B1grads,1,lenBias1,learningRate);
        update(C, Cgrads, lenAlpha,dimensions,learningRate);
        free(W2grads);free(W1grads);free(B1grads);free(B2grads);
        free(Egrads);free(Cgrads);
        free(H);free(logits);free(probs);free(dLdYp);free(dhdz);
        free(W2T);free(W1T);free(dldh);free(dldz);
    }
    printf("Final loss after training is %7.4f\n",loss);
    printf("The final learning rate is %5.3lf\n",learningRate);
    loss=splitLoss(text4Train,C,W1,B1,W2,B2,alphabet,blocksize,dimensions,inNodes,
                hidNodes,outNodes,lenBias1,lenBias2);
    printf("Loss for Train is %7.4lf\n",loss);
    loss=splitLoss(text4Dev,C,W1,B1,W2,B2,alphabet,blocksize,dimensions,inNodes,
                hidNodes,outNodes,lenBias1,lenBias2);
    printf("Loss for Dev   is %7.4lf\n",loss);
    //loss=splitLoss(text4Test,C,W1,B1,W2,B2,alphabet,blocksize,dimensions,inNodes,
      //          hidNodes,outNodes,lenBias1,lenBias2);
    //printf("Loss for Test  is %7.4lf\n",loss);
    makeMore(C,W1,B1,W2,B2,20,alphabet,blocksize,dimensions,
            inNodes,hidNodes,outNodes,lenBias1,lenBias2);
    return 0;
}
