#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>

#include "helpers.h"

char * getNames(const char *fileName){
    // get names from file and place into a string
    struct stat st;
    int fd = open(fileName,O_RDONLY);
    if(fd == -1) {
        perror("Error: ");
        exit(errno);
    }
    stat(fileName, &st);
    off_t size = st.st_size;
    // need room for \n at front and \0 at end
    char *names=(char *)calloc(size+2,sizeof(char));
    names[0]='\n';
    int num=read(fd,names+1,size);
    if(num < 0) {
        perror("Error: ");
        exit(errno);
    }
    close(fd);
    // change new line characters to periods
    int len = strlen(names);
    for(int i=0; i<len; i++)
        if(names[i] == '\n')names[i] = '.';
    return names;
}
int getNumNames(char *names){
    int len=strlen(names);
    int count=0;
    for(int i=1;i<len;i++)
        if(names[i] == '.') count++;
    return count;
}
char *getOneName(int n,char *names){
    // return one name without the periods
    // n==3 will return the 3rd name
    int i=1;
    char *temp=(char *)calloc(strlen(names)+1,sizeof(char));
    strcpy(temp,names);
    char *oneName=strtok(temp+1,".");
    while(oneName != NULL){
        if(i >= n) break;
        oneName=strtok(NULL,".");
        i++;
    }
    char *theName=(char *)calloc(strlen(oneName)+1,sizeof(char));
    strcpy(theName,oneName);
    free(temp);
    return theName;
}
char *getSomeNames(char *text, int howMany){
    // words are deliminated by periods
    // return a string of words starting from the
    // beginning of the text string
    int to=0,count=0;
    int len=strlen(text);
    for(int i=1;i<len;i++){
        if(text[i] == '.'){
            count++;
            if(count >= howMany){
                to = i;
                break;
            }
        }
    }
    char *names = (char *)calloc(++to,sizeof(char));
    strncpy(names,text,to);
    return names;
}
char *getSetOfNames(char *text, int fm, int to){
    // return a string of names from (fm) to (to)
    int start=0,stop=0,count=0;
    int len=strlen(text);
    for(int i=0;i<len;i++){
        if(text[i] == '.'){
            count += 1;
            if(count == fm){
                start = i;
            }else{
                if(count-1 == to)
                    stop = i;
            }
        }
    }
    char *names = (char *)calloc(stop-start+2,1);
    strncpy(names,text+start,stop-start+1);
    return names;
}
float *makeTable(gsl_rng *pr,int rows,int cols){
    float *C=(float *)calloc(rows*cols,sizeof(float));
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            *(C+i*cols+j) = gsl_ran_gaussian(pr,1.0); 
        // normal distribution - standard deviation of 1.0
    return C;
}
char *getAlphabet(char *text){
    // get the list of unique characters from text string
    // assume ASCII so need space for max of 128 characters
    char chars[128] = {0};
    char *letters = (char *)calloc(128,1);
    int len = strlen(text);
    for(int i=0;i<len;i++)
        chars[(int)text[i]] = text[i];
    for(int i=0,j=0;i<128;i++)
        // append characters to letters string
        if(chars[i] != 0)
            letters[j++] = chars[i];
    return letters;
}
int getIndex(char *s, char c){
    // returns the index of c in s
    int len=strlen(s);
    for(int i=0;i<len;i++)
        if(s[i] == c) return i;
    return -1;
}
