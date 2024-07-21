#ifndef FUNCTIONS_H 
#define FUNCTIONS_H 

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>

int     getIndex(char *s, char c);
float *makeTable(gsl_rng *pr,int rows,int cols);

#endif
