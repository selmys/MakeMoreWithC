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
    stat(fileName, &st);
    off_t size = st.st_size;
    // need room for \n at front and \0 at end
    char *names=calloc(size+2,sizeof(char));
    names[0]='\n';
    read(fd,names+1,size);
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
    char *letters = calloc(128,1);
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
int *getBiGramCounts(int *intList,int intLen,int lenAlpha){
    // w is the array of weights (27x27) integers
    // each element contains the bigram count
    // . a b c d e f g .... z
    // a
    // b
    // c
    // d
    // etc
    // for example column 'a', row 'd' contains the
    // number of times the bigram 'ad' occured in
    // the array
    int *w = calloc(intLen*intLen,4);
    for(int i=0,r,c;i<intLen;i++){
        r=*(intList+i);
        c=*(intList+i+1);
        //printf("%3d%3d\n",r,c);
        (*(w+r*lenAlpha + c))++;
    }
    return w;
}
double *convertCounts2Prob(int *counts, int lenAlpha){
    double *probs = calloc(lenAlpha*lenAlpha,sizeof(double));
    // for each row in counts find the sum
    // for each element in the row divide its value by the sum
    // and place into corresponding element of WP
    // now the sum of probabilities of each row should equal 1
    // WP array is normalized
    float sum;
    for(int i=0;i<lenAlpha;i++){
        sum=0.0;
        for(int j=0;j<lenAlpha;j++)
            sum += *(counts+i*lenAlpha+j);
        for(int j=0;j<lenAlpha;j++)
            *(probs+i*lenAlpha+j) = *(counts+i*lenAlpha+j) / sum;
    }
    return probs;
}
float getLoss(int *encList, int listLen, double *probs, int lenAlpha){
    // now compute the log likelihood (-inf to zero)
    // we do this by adding the logs of all the probabilities
    // because log(a*b*c) = log(a)+log(b)+log(c)
    float log_likelihood=0.0; // sum of logs of all probabilities
    for(int i=0;i<listLen;i++)
        log_likelihood += log(*(probs+encList[i]*lenAlpha+encList[i+1]));
    printf("The log likelihood is %9.4lf\n",log_likelihood);
    // now get the negative log likelihood
    float nll = -log_likelihood;
    printf("The negative loglikelihood is %9.4lf\n",nll);
    printf("The number of unique bigrams is %d\n",listLen);
    // and the normalized loss is nll/count
    printf("The normalized (average or loss function) nll/count is %lf\n",nll/listLen);
    return nll/listLen;
}
void smoothCounts(int *counts, int length){
	// add 1 to each count
	for(int i=0;i<length;i++)
		counts[i] = counts[i] + 1;
	return;
}
void makeMore(double *WP, char *alphabet, int count){
    // let's make count more names
    // but first we need to set up the RNG
    // using GSL (GNU Scientific Library)
    // first select the rng -random number generator- to use
    gsl_rng *pr = gsl_rng_alloc(gsl_rng_ran0);
    // then set the seed to the rng to the time
    //gsl_rng_set(pr, 2147483647); // same as in Andrej's video
    int lenAlpha = strlen(alphabet);
    // n is a vector of integers of length lenAlpha
    unsigned int *n = calloc(lenAlpha,sizeof(int));
    // and start generating count number of words
    int row=0;// pick a random letter starting from row 0
    for(int i=0;i<count;i++){
        // keep going until we hit an end of word (zero)
        while (1){
            gsl_ran_multinomial(pr,lenAlpha,1,WP+row*lenAlpha,n);
            // find which element of n has a 1
            for(int j=0;j<lenAlpha;j++){
                if(n[j]!=0) row=j;
            }
            // print out the character
            if(row != 0)
                printf("%c",alphabet[row]);
            else
                break;
        }
        printf(".\n");
    }
    return;
}
void saveBiGramCounts(int *w, int n, char *file){
    // file is used by gnuplot
    FILE *fp = fopen(file,"w");
    for(int i=0;i<=n;i++)
        if(i==0)
            fprintf(fp,"XXX;");
        else
            if(i<n)
                fprintf(fp,"%c;",95+i); // print column headers ' a b c ...
            else
                fprintf(fp,"%c",95+i); // print column headers ' a b c ...
    fprintf(fp,"\n");
    for(int i=0;i<n;i++){
        fprintf(fp,"%c;",96+i); // print the row headers a b c ...
        for(int j=0;j<n;j++){
            if(j<n-1)
                fprintf(fp,"%d;",*(w+i*n+j));
            else
                fprintf(fp,"%d",*(w+i*n+j));
        }
        fprintf(fp,"\n");
    }
    fprintf(fp,"\n");
    fflush(fp);
    fclose(fp);
    return;
}
int main(){
    char *names = getNames("names.txt");
    char *alphabet = getAlphabet(names);
	int numBigrams = strlen(names) - 1; // > 200k
	int lenAlpha = strlen(alphabet); // 27
    int *encList = encode(names,alphabet);
    int *counts = getBiGramCounts(encList,strlen(names)-1,strlen(alphabet));
	smoothCounts(counts,numBigrams*lenAlpha);
	saveBiGramCounts(counts,lenAlpha,"bigramcounts.txt");
    double *probs = convertCounts2Prob(counts,strlen(alphabet));
    printf("the loss function is %g\n", getLoss(encList,strlen(names)-1,probs,strlen(alphabet)));
    makeMore(probs,alphabet,20);
    return 0;
}
