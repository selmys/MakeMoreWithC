#ifndef BYTEPAIR_H
#define BYTEPAIR_H

int getNumTokens(char *file);
char **getTokens(char *file,int numTokens);
int getVocabLen(char **tokens,int numTokens);
int *encodeTokens(char **tokens,int numTokens,char **vocab,int vocabLen);
int *getRawBytes(char *file,int fileLen);
int *getAlphabet(int *line, int lineLen, int *lenAlpha);
int removeDuplicates(int *alphabet, int lenAlpha);
int getMaxAlpha(int *alphabet, int lenAlpha);
char **getVocab(char **tokens,int numTokens,int vocabLen);
void encodeRawData(int *rawBytes,int numRawBytes,int *alphabet);
void unEcodeRawData(int *rawBytes,int numRawBytes,int *vocab);
int getBytePair(int line[],int lineSize,int lenAlpha,int *n1,int *n2,int *max);
int *updateVocab(int *vocab,int vocabLen,int nextValue);
void updateEncodedTokens(int encodedTokens[],int numTokens,int history[4]);
int *undoHistory(int history[][4],int *line,int *len,int numTokens);

#endif