#include <string.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>

#include "helpers.h"

float *makeTable(gsl_rng *pr,int rows,int cols){
    float *C=(float *)calloc(rows*cols,sizeof(float));
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            *(C+i*cols+j) = gsl_ran_gaussian(pr,1.0); 
        // normal distribution - standard deviation of 1.0
    return C;
}
int getIndex(char *s, char c){
    // returns the index of c in s
    int len=strlen(s);
    for(int i=0;i<len;i++)
        if(s[i] == c) return i;
    return -1;
}
