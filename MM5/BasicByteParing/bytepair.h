#ifndef BYTEPAIR_H
#define BYTEPAIR_H

int getFileSize(char *file);
int *getRawBytes(char *file,int fileLen);
int *getAlphabet(int *line, int lineLen, int *lenAlpha);
int removeDuplicates(int *alphabet, int lenAlpha);
int getMaxAlpha(int *alphabet, int lenAlpha);
int *getVocab(int *alphabet,int lenAlpha,int lenVocab);
void encodeRawData(int *rawBytes,int numRawBytes,int *alphabet,int lenAlpha);
void unEncodeRawData(int *rawBytes,int numRawBytes,int *alphabet);
int getBytePair(int line[],int lineSize,int *alphabet,int lenAlpha,int *n1,int *n2,int *max);
int *updateAlphabet(int *alphabet,int lenAlpha,int nextValue);
void updateLine(int line[],int numRawBytes,int history[4]);
int undoHistory(int history[][4],int *line,int len);
int *undoHistory1(int history[][4],int *line,int *len,int numRawBytes);

#endif