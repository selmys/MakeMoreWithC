#ifndef FUNCTIONS_H 
#define FUNCTIONS_H 

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>

char   *getAlphabet(char *text);
int     getIndex(char *s, char c);
char   *getNames(const char *fileName);
int     getNumNames(char *names);
char   *getOneName(int n,char *names);
char   *getSetOfNames(char *text, int fm, int to);
char   *getSomeNames(char *text, int howMany);
float *makeTable(gsl_rng *pr,int rows,int cols);

#endif
