#include <stdio.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
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
int *getVocab(int *alphabet,int lenAlpha,int lenVocab){
    int *vocab=(int *)calloc(lenVocab,sizeof(int));
    for(int i=0;i<lenAlpha;i++){
        vocab[alphabet[i]] = i;
    }
    return vocab;
}
void encodeRawData(int *rawBytes,int numRawBytes,int *alphabet,int lenAlpha){
    for(int i=0;i<numRawBytes;i++){
        for(int j=0;j<lenAlpha;j++){
            if(rawBytes[i] == alphabet[j]){
                rawBytes[i] = j;
                break;
            }
        }
    }
    return;
}
void unEncodeRawData(int *rawBytes,int numRawBytes,int *alphabet){
    for(int i=0;i<numRawBytes;i++){
        rawBytes[i] = alphabet[rawBytes[i]];
    }
    return;
}
int getBytePair(int *line,int lineSize,int *alphabet,int lenAlpha,int *n1,int *n2,int *max){
	// get the most frequent pair of bytes (integers) in the line
	// start by making a square array of lenAlpha
    
	int *table=(int *)calloc(lenAlpha*lenAlpha,sizeof(int));
	// fill table with byte pair counts
    int previous=0; // to avoid double counting pairs
	for(int i=0;i<lineSize-1;i++){
        for(int j=0;j<lenAlpha;j++){
            previous=0;
            if(line[i+0]==alphabet[j]){ 
                //previous=j;
                *n1=j;
                for(int k=0;k<lenAlpha;k++){
                    if(line[i+1]==alphabet[k]){ 
                        previous=k;
                        *n2=k;
                        break;
                    }
                }
                break;
            }   
        }
		if(*n1 != *n2)
            *(table+*n1*lenAlpha+*n2) += 1;
        else if(*n1 != previous)
            *(table+*n1*lenAlpha+*n2) += 1;
    }
	// find byte pair with maximum count
	*max = 1;
	for(int i=0;i<lenAlpha;i++)
		for(int j=0;j<lenAlpha;j++)
			if(*(table+i*lenAlpha+j) > *max){
				*max=*(table+i*lenAlpha+j);
				*n1=i;
				*n2=j;
			}
	if(*max > 1){
		lenAlpha += 1;
	}
    free(table);
	return lenAlpha;
}
int *updateAlphabet(int *alphabet,int lenAlpha,int nextValue){
    int *newAlphabet=(int *)calloc(lenAlpha,sizeof(int));
    // copy alphabet to newAlphabet
    int i;
    for(i=0;i<lenAlpha-1;i++)
        newAlphabet[i] = alphabet[i];
    // add next value to end
    newAlphabet[i] = nextValue;
    free(alphabet);
    return newAlphabet;    
}
void updateLine(int *line,int numRawBytes,int history[4]){
	// every time we do a merge, the length of line is reduced
    int *newLine=(int *)calloc(numRawBytes-history[0],sizeof(int));
	int skip=0,j=0,i;
    for(i=0;i<numRawBytes-1;i++){
	    if(!skip){
            if(line[i]==history[2] && line[i+1]==history[3]){
    			newLine[j++]=history[1];
                skip=1;
    		}else{
                newLine[j++]=line[i];
            }
    	}else{
            skip=0;
        }
    }
    if(!skip){
        newLine[j++]=line[i];
    }
    // copy newLine to line
    for(int i=0;i<j;i++){
        line[i]=newLine[i];
    }
    free(newLine);
	return;
}
int undoHistory(int history[][4],int *line,int len){
	// step through history and count how many changes are 
	// needed to make in the line
	int numChanges=0;
	for(int i=0;i<MERGES;i++){
		numChanges += history[i][0];
	}
	// make room for newLine (size of original data)
	int *newLine=(int *)calloc(100*len+numChanges,sizeof(int));
	// go backwards through history expanding each bytepair
	int k;
	for(int i=MERGES-1;i>=0;i--){
		k=0;
      //  printf("len is %d\n",len);
		for(int j=0;j<len;j++){
			if(line[j] != history[i][1]){
				newLine[k++] = line[j];
			}else{
                //fprintf(stderr,"Undoing %d as %d and %d\n",
                 //   history[i][1],history[i][2],history[i][3]);
				newLine[k++] = history[i][2];
				newLine[k++] = history[i][3];
			}
		}
		// copy newLine to Line and update length
		for(int h=0;h<k;h++)
			line[h] = newLine[h];
        len = k;
	}
	return len;
}
int *undoHistory1(int history[][4],int *line,int *len,int numTokens){
	// make room for newLine (size of original data)
    printf("numTokens is %d\n",numTokens);
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
               // printf("skipping %d\n",newLine[j]);
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
        for(int n=0;n<k;n++)
            newLine[n] = nextLine[n];
	}
    int *results=(int *)calloc(*len,sizeof(int));
    for(int i=0;i<*len;i++)
        results[i] = newLine[i];
    free(newLine);
    free(nextLine);
	return results;
}