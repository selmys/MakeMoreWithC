#include <stdio.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <pcre.h>
#define OFFSETCOUNT 6    // (capturing_group_count + 1) * 3
#include "params.h"
#include "bytepair.h"

int getFileSize(char *file){
    struct stat st;
    int fd = open(file,O_RDONLY);
    if(fd == -1) {
        perror("Error: ");
        exit(errno);
    }
    stat(file, &st);
    off_t size = st.st_size;
	close(fd);
	return size;
}
int getNumTokens(char *fileName) {
    const char *error;
    int erroffset;
    int offsetcount;
    int offsets[OFFSETCOUNT];
    const char *result;

    int subject_size = getFileSize(fileName); 
    printf("Subject Size is %d\n",subject_size);
    char *src=(char *)calloc(subject_size+1,sizeof(char));
    FILE *fp;
    fp=fopen(fileName,"r");
    for(int i=0;i<subject_size;i++)
	    fscanf(fp,"%c",&src[i]);
    fclose(fp);

    char pattern[] = "('s|'t|'re|'ve|'m|'ll|'d|[ \n]?[a-zA-Z]+|[ \n]?[0-9]+|[ \n]?[^ \na-zA-Z0-9]+|[ \n]+(?![^ \n])|[ \n]+)";
    
    pcre *re = pcre_compile(pattern, 0, &error, &erroffset, NULL);
    if (re == NULL) {
        printf("PCRE compilation failed at offset %d: %s/n", erroffset, error);
        return 1;
    }

    int numTokens=0;
    offsets[1] = 0;
    while ((offsetcount = pcre_exec(re, NULL, src, strlen(src), offsets[1], 0, offsets, OFFSETCOUNT)) >= 0) { 
        if (pcre_get_substring(src, offsets, offsetcount, 1, &result) >= 0) {
            printf("#%s#\n", result);
            numTokens++;
            pcre_free_substring(result);
        }
    } 
    
    free(re);
    return numTokens;
}
char **getTokens(char *fileName,int numTokens){
    const char *error;
    int erroffset;
    int offsetcount;
    int offsets[OFFSETCOUNT];
    const char *result;

    int subject_size = getFileSize(fileName); 
    printf("Subject Size is %d\n",subject_size);
    char *src=(char *)calloc(subject_size+1,sizeof(char));
    FILE *fp;
    fp=fopen(fileName,"r");
    for(int i=0;i<subject_size;i++)
	    fscanf(fp,"%c",&src[i]);
    fclose(fp);

    char pattern[] = "('s|'t|'re|'ve|'m|'ll|'d|[ \n]?[a-zA-Z]+|[ \n]?[0-9]+|[ \n]?[^ \na-zA-Z0-9]+|[ \n]+(?![^ \n])|[ \n]+)";
    char **tokens=NULL;
    pcre *re = pcre_compile(pattern, 0, &error, &erroffset, NULL);
    if (re == NULL) {
        printf("PCRE compilation failed at offset %d: %s/n", erroffset, error);
        return tokens;
    }

    tokens=(char **)calloc(numTokens,sizeof(char *));
    int i=0;
    offsets[1] = 0;
    while ((offsetcount = pcre_exec(re, NULL, src, strlen(src), offsets[1], 0, offsets, OFFSETCOUNT)) >= 0) { 
        if (pcre_get_substring(src, offsets, offsetcount, 1, &result) >= 0) {
            //printf("#%s#\n", result);
            tokens[i] = (char *)calloc(strlen(result+1),sizeof(char));
            strcpy(tokens[i++],result);
            pcre_free_substring(result);
        }
    } 
    free(re);
    return tokens;
}
int getVocabLen(char **tokens,int numTokens){
    int count=0;
    int j;
    for(int i=0;i<numTokens;i++){
        for(j=i+1;j<numTokens;j++)
            if(strcmp(tokens[i],tokens[j]) == 0)
                break;
        if(j==numTokens)
            count++;
    }
    return count;
}
char **getVocab(char **tokens,int numTokens,int vocabLen){
    char **vocab=NULL;
    vocab=(char **)calloc(vocabLen,sizeof(char *));
    int j,k=0;
    for(int i=0;i<numTokens;i++){
        for(j=i+1;j<numTokens;j++)
            if(strcmp(tokens[i],tokens[j]) == 0)
                break;
        if(j==numTokens){
            vocab[k]=(char *)calloc(strlen(tokens[i]+1),sizeof(char));
            strcpy(vocab[k],tokens[i]);
            k++;
        }
    }
    return vocab;
}
int *encodeTokens(char **tokens,int numTokens,char **vocab,int vocabLen){
    int *encodedTokens=(int*)calloc(numTokens,sizeof(int));
    for(int i=0;i<numTokens;i++){
        for(int j=0;j<vocabLen;j++){
            if(strcmp(tokens[i],vocab[j]) == 0){
                encodedTokens[i] = j;
            }
        }
    }    
    return encodedTokens;
}
int *getRawBytes(char *file, int fileLen){
	int *bytes=(int *)calloc(fileLen,sizeof(int));
	FILE *fp;
	fp=fopen(file,"r");
    unsigned char c;
    int len=0;
	c = fgetc(fp);
    while(!feof(fp)){
        bytes[len++]=c;
		c=fgetc(fp);
	}
    fclose(fp);
    return bytes;
}
int *getAlphabet(int *line, int lineLen, int *lenAlpha){
    // data is raw bytes so max alphabet size is 256
    int bytes[256] = {0};
    for(int i=0;i<lineLen;i++)
        if(bytes[line[i]] == 0)
            bytes[line[i]] = line[i];
    *lenAlpha=0;
    // count unique bytes
    for(int i=0;i<256;i++)
        if(bytes[i] != 0)
            *lenAlpha = *lenAlpha + 1;
    // allocate space for alphabet
    int *alphabet=(int *)calloc(*lenAlpha,sizeof(int));
    // fill alphabet from bytes
    int j=0;
    for(int i=0;i<256;i++)
        if(bytes[i] != 0){
            alphabet[j++] = bytes[i];
        }
    return alphabet;
}
int removeDuplicates(int *alphabet, int lenAlpha){
    for(int i=0,j=0;i<lenAlpha;i++){
        if(alphabet[++i] != alphabet[j]){
            if(i != ++j){
                alphabet[j] = alphabet[i];
                lenAlpha--;
            }
        }
    }
    return lenAlpha;
}
int getMaxAlpha(int *alphabet, int lenAlpha){
    int max=alphabet[0];
        for(int i=1;i<lenAlpha;i++)
            if(alphabet[i] > max)
                max = alphabet[i];
    return max;
}
void encodeRawData(int *rawBytes,int numRawBytes,int *vocab){
    for(int i=0;i<numRawBytes;i++){
        rawBytes[i] = vocab[rawBytes[i]];
    }
    return;
}
void unEcodeRawData(int *rawBytes,int numRawBytes,int *alphabet){
    for(int i=0;i<numRawBytes;i++){
        rawBytes[i] = alphabet[rawBytes[i]];
    }
    return;
}
int getBytePair(int *encodedTokens,int numTokens,int vocabLen,int *n1,int *n2,int *max){
	// get the most frequent pair of bytes (integers) in the encodedTokens
	// start by making a square array of vocabLen   
	int *table=(int *)calloc(vocabLen*vocabLen,sizeof(int));
	// fill table with byte pair counts
	for(int i=0;i<numTokens-1;i++){
        *n1=encodedTokens[i];
        *n2=encodedTokens[i+1];
       // printf("(%d,%d) ",*n1,*n2);
        *(table + (*n1)*vocabLen + (*n2)) += 1;
    }
	// find byte pair with maximum count
	*max = 0;
	for(int i=0;i<vocabLen;i++)
		for(int j=0;j<vocabLen;j++)
			if(*(table+i*vocabLen+j) > *max){
				*max=*(table+i*vocabLen+j);
				*n1=i;
				*n2=j;
			}
	if(*max > 1){
		vocabLen += 1;
	}
    free(table);
	return vocabLen;
}
int *updateVocab(int *vocab,int vocabLen,int nextValue){
    int *newVocab=(int *)calloc(vocabLen,sizeof(int));
    // copy vocab to newVocab
    int i;
    for(i=0;i<vocabLen-1;i++)
        newVocab[i] = vocab[i];
    // add next value to end
    newVocab[i] = nextValue;
    free(vocab);
    return newVocab;    
}
void updateEncodedTokens(int *encodedTokens,int numTokens,int history[4]){
	// every time we do a merge, the length of line is reduced
    int *newTokens=(int *)calloc(numTokens-history[0],sizeof(int));
	int skip=0,j=0,i;
    for(i=0;i<numTokens-1;i++){
	    if(!skip){
            if(encodedTokens[i]==history[2] && encodedTokens[i+1]==history[3]){
    			newTokens[j++]=history[1];
                skip=1;
    		}else{
                newTokens[j++]=encodedTokens[i];
            }
    	}else{
            skip=0;
        }
    }
    if(!skip){
        newTokens[j++]=encodedTokens[i];
    }
    // copy newTokens to encodedTokens
    for(int i=0;i<j;i++){
        encodedTokens[i]=newTokens[i];
    }
    free(newTokens);
	return;
}
int *undoHistory(int history[][4],int *line,int *len,int numTokens){
	// make room for newLine (size of original data)
	int *newLine=(int *)calloc(numTokens,sizeof(int));
	int *nextLine=(int *)calloc(numTokens,sizeof(int));
    // copy line to newLine
    for(int i=0;i<*len;i++){
        newLine[i] = line[i];
    }
	// go backwards through history expanding each bytepair
	int k;
	for(int i=MERGES-1;i>=0;i--){
		k=0;
		for(int j=0;j<*len;j++){
			if(newLine[j] != history[i][1]){
                //printf("skipping %d\n",newLine[j]);
                nextLine[k++] = newLine[j];
			}else{
                printf("Undoing %d as %d and %d\n",
                    history[i][1],history[i][2],history[i][3]);
				nextLine[k++] = history[i][2];
				nextLine[k++] = history[i][3];
			}
		}
        *len = k;
        // copy nextLine to newLine
        for(int k=0;k<*len;k++)
            newLine[k] = nextLine[k];
	}
    int *results=(int *)calloc(*len,sizeof(int));
    for(int i=0;i<*len;i++)
        results[i] = newLine[i];
    free(newLine);
    free(nextLine);
	return results;
}