#ifndef PRINTERS_H
#define PRINTERS_H

void    printBigrams(char *names);
void    print(const char *title, float *matrix, int rows, int cols);
void    printE(const char *title, float *matrix, int rows, int cols);
void    printEmb(float *emb, int l, int w, int h);
void    printEncodedBigrams(int *X,int *Y);
void    printInt(const char *title, int *matrix, int rows, int cols);

#endif