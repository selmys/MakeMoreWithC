#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>
#include <math.h>
char * getNames(char *fileName){
    // get names from file and place into a string
    struct stat st;
    int fd = open(fileName,O_RDONLY);
    if(!fd){
    	printf("Cannot open file!\n");
    	exit(2);
    }
    stat(fileName, &st);
    off_t size = st.st_size;
    // need room for \n at front and \0 at end
    char *names=calloc(size+2,sizeof(char));
    names[0]='\n';
    int n=read(fd,names+1,size);
    if(n <= 0){
    	printf("Cannot read file!\n");
    	close(fd);
    	exit(2);
    }
    close(fd);
    // change new line characters to periods
    int len = strlen(names);
    for(int i=0; i<len; i++)
        if(names[i] == '\n')names[i] = '.';
    return names;
}
char *getAlphabet(char *text){
    // get the list of unique characters from text string
    // assume ASCII so need space for max of 128 characters
    char chars[128] = {0};
    char *letters = calloc(128,sizeof(char));
    int len = strlen(text);
    for(int i=0;i<len;i++)
        chars[(int)text[i]] = text[i];
    for(int i=0,j=0;i<128;i++)
        // append characters to letters string
        if(chars[i] != 0)
            letters[j++] = chars[i];
    return letters;
}
int *encode(char *text, char *alphabet){
    // encode text characters into integers based on
    // their positions in the alphabet string
    int lenText = strlen(text);
    int lenAlph = strlen(alphabet);
    int *encList = calloc(lenText,sizeof(int));
    for(int i=0;i<lenText;i++)
        for(int j=0;j<lenAlph;j++)
            if(text[i] == alphabet[j]){
                encList[i] = j;
                break;
            }
    return encList;
}
void makeXY(int **xs,int **ys,int *encList,int numBiGrams){
    *xs = calloc(numBiGrams,sizeof(int));
    *ys = calloc(numBiGrams,sizeof(int));
    for(int i=0;i<numBiGrams;i++){
        *(*xs+i) = encList[i];
        *(*ys+i) = encList[i+1];
    }
    return;
}
float *oneHotEncode(int *s, int numRows, int rowLen){
    //s is the list of integers to encode
    //in our example we should get an array numBiGrams X lenAlphabet
    //containing only 1's and 0's
    //this is known as one-hot encoding
    float *xy=calloc(numRows*rowLen,sizeof(float));
    for(int i=0;i<numRows;i++){
        for(int j=0;j<rowLen;j++) {
            if(j == *(s+i)){
                *(xy+i*rowLen+j) = 1;
            }
        }
    }
    return xy;
}
void printOneHotEncode(int *s, int numRows, int rowLen){
    //s is the list of integers to encode
    //in our example we should get an array numBiGrams X lenAlphabet
    //containing only 1's and 0's
    //this is known as one-hot encoding
    printf("XXX;.;a;b;c;d;e;f;g;h;i;j;k;l;m;n;o;p;q;r;s;t;u;v;w;x;y;z\n");
    for(int i=0;i<numRows;i++){
    	printf("%d;",i);
        for(int j=0;j<rowLen;j++) {
            if(j == *(s+i)){
                if(j == rowLen-1){printf("1");}else{printf("1;");} 
            }else{
            	if(j == rowLen-1){printf("0");}else{printf("0;");}
            }
        }
        printf("\n");
    }
    return;
}
double *probs2d(float *probs, int len){
	// convert floats to doubles
    double *p=(double *)calloc(len,sizeof(double));
    for(int i=0;i<len;i++)
        p[i]=probs[i];
    return p;
}
float *makeTable(gsl_rng *pr,int rows,int cols){
    float *C=(float *)calloc(rows*cols,sizeof(float));
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            *(C+i*cols+j) = gsl_ran_gaussian(pr,1.0); 
        // normal distribution - standard deviation of 1.0
    return C;
}
void print(char *title, float *matrix, int rows, int cols){
    printf("========== %s ===========\n",title);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++)
            printf("%7.4lf ",*(matrix+i*cols+j));
        printf("\n");
    }
    return;
}
float *multAB(float *A, float *B, int ai, int aj, int bi, int bj) {
    // matrix multiplication C=AxB iff aj==bi
    float sum;
    float *C = calloc(ai*bj,sizeof(float));
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
float *getCounts(float *logits, int rows, int cols){
    // change log counts to counts
    float *counts = calloc(rows*cols,sizeof(float));
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            *(counts+i*cols+j) = exp(*(logits+i*cols+j));
    return counts;
}
float *getProbs(float *counts, int rows, int cols){
    // change counts to probabilities by normalizing each row
    float *probs=calloc(rows*cols,sizeof(float));
    float sum;
    for(int i=0;i<rows;i++){
        sum=0.0;
        for(int j=0;j<cols;j++)
            sum += *(counts+i*cols+j);
        for(int j=0;j<cols;j++)
            *(probs+i*cols+j) = *(counts+i*cols+j)/sum;
    }
    return probs;
}
float *softMax(float *logits,int rows,int cols){
    float *counts=getCounts(logits,rows,cols);
    float *probs=getProbs(counts,rows,cols);
    free(counts);
    return probs;
}
float getLoss(float *probs, int rows, int cols, int *xs, int *ys){
    // in first example rows=5,cols=27,xs=0 5 13 13 1, ys=5 13 13 1 0
    float *nlls=calloc(rows,sizeof(float));
    for(int i=0; i<rows; i++)
        nlls[i] = -log(*(probs+i*cols+*(ys+i)));
    float loss=0.0;
    for(int i=0;i<rows;i++)
        loss += nlls[i];
    free(nlls);
    return loss/rows;
}
float *transpose(float *a,int r, int c){
    // transpose a matrix
    float *t = calloc(r*c,sizeof(float));
    for (int i=0;i<r;i++) {
		for (int j=0;j<c;j++)
			*(t+j*r+i) = *(a+i*c+j);
	}
    return t;
}
float *getGradients(float *probs,int numBigrams,int lenAlpha,int *xs,int *ys){
    // in first example rows=5,cols=27,xs=0 5 13 13 1, ys=5 13 13 1 0
    float *dLdy = calloc(numBigrams*lenAlpha,sizeof(float));
    memcpy(dLdy,probs,numBigrams*lenAlpha*sizeof(float));
    // get Y-Yp
    for(int i=0; i<numBigrams; i++)
        for(int j=0;j<lenAlpha;j++)
            if(ys[i] == j)
                *(dLdy+i*lenAlpha+j) = *(probs+i*lenAlpha+j) - 1.0;
    float *X = oneHotEncode(xs,numBigrams,lenAlpha);
    // transpose X
    float *XT = transpose(X,numBigrams,lenAlpha);
    // get dLdw = XT * dLdy
    float *grads=multAB(XT,dLdy,lenAlpha,numBigrams,numBigrams,lenAlpha);
    // divide by numBigrams
    for(int i=0;i<lenAlpha*lenAlpha;i++)
        grads[i] /= numBigrams;
    free(dLdy);free(X);free(XT);
    return grads;
}
void updateW(float *w, float *grads, int lenAlpha){
    // update the weights
    float learningRate = -50;
    for(int i=0;i<lenAlpha;i++)
        for(int j=0;j<lenAlpha;j++)
            *(w+i*lenAlpha+j) = *(w+i*lenAlpha+j) + learningRate * *(grads+i*lenAlpha+j);
    return;
}
void makeMore(gsl_rng *pr, float *W, char *alphabet, int count){
    int lenAlpha = strlen(alphabet);
    // n is a vector of integers of length lenAlpha
    unsigned int *n = calloc(lenAlpha,sizeof(int));
    // and start generating count number of words
    float *xenc,*logits,*probs;
    double *probsd;
    int row = 0;// pick a random letter starting from row 0
    for(int i=0;i<count;i++){
        // keep going until we hit an end of word (zero)
        while (1){
            xenc = oneHotEncode(&row,1,lenAlpha);
            logits = multAB(xenc,W,1,lenAlpha,lenAlpha,lenAlpha);
            probs = softMax(logits, 1, lenAlpha);
            probsd = probs2d(probs,lenAlpha);
            gsl_ran_multinomial(pr,lenAlpha,1,probsd,n);
            // find which element of n has a 1
            for(int j=0;j<lenAlpha;j++){
                if(n[j]!=0) row=j;
            }
            // print out the character
            if(row != 0){
                printf("%c",alphabet[row]);
                free(xenc);free(logits);free(probs);
            }
            else
                break;
        }
        printf(".\n");
        free(xenc);free(logits);free(probs);
    }
    return;
}
int main(){
	// we'll have to use GNU Scientific Library
    // first select the rng -random number generator
    // used to get random weights and biases
    gsl_rng *pr = gsl_rng_alloc(gsl_rng_mt19937);
    srand(42); // initialize rng - used to get random minibatch
    char *names=getNames("names.txt");
    char *alphabet = getAlphabet(names);
    int lenAlpha=strlen(alphabet);
    int numBigrams=strlen(names)-1;
    printf("lenAlpha is %d and numBigrams is %d\n",lenAlpha,numBigrams);
    // encode all the names
    int *encList = encode(names,alphabet);
    int *xs; // inputs
    int *ys; // expected outputs (labels)
    makeXY(&xs,&ys,encList,numBigrams);
    //printOneHotEncode(xs,20,27);
    //exit(1);
    float *xenc=oneHotEncode(xs,numBigrams,lenAlpha);
    float *weights=makeTable(pr,lenAlpha,lenAlpha);
    //print("weights",weights,27,27);
    //exit(1);
    float *logits,*probs,*grads;
    for(int i=0;i<150;i++){
        // get the log counts (logits) using matrix multiplication
        logits=multAB(xenc,weights,numBigrams,lenAlpha,lenAlpha,lenAlpha);
        // convert to probabilities
        probs=softMax(logits, numBigrams, lenAlpha);
        // get the loss
        printf("Loss = %7.4lf\n",getLoss(probs,numBigrams,lenAlpha,xs,ys));
        // do backward propagation to get the gradients of the weights
        grads = getGradients(probs,numBigrams,lenAlpha,xs,ys);
        // update the weights
        updateW(weights,grads,lenAlpha);
        free(logits);free(probs);free(grads);
    }
    makeMore(pr,weights,alphabet,20);
    return 0;
}
