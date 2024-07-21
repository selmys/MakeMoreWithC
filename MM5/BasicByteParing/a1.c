#include <stdio.h>
#include <stdlib.h>

int getBytePair(int line[],int lineSize,int tableSize,int *n1,int *n2,int *max){
	// get the most frequent pair of integers in the line
	int *table=(int *)calloc(tableSize*tableSize,sizeof(int));	
	for(int i=0;i<lineSize-1;i++)
		*(table+line[i]*tableSize+line[i+1]) += 1;
	int row=-1;
	int col=-1;
	for(int i=0;i<tableSize;i++)
		for(int j=0;j<tableSize;j++)
			if(*(table+i*tableSize+j) > *max){
				*max=*(table+i*tableSize+j);
				row=i;
				col=j;
			}
	if(*max < 2) return tableSize;
	*n1=row;
	*n2=col;
	tableSize += 1;
	return tableSize;
}
int updateLine(int line[],int len,int nextNum,int n1,int n2){
	int *newLine=(int *)calloc(len,sizeof(int));
	int newLen=len;
	int skip=0;
	int i;
	int j=0;
	for(i=0;i<len-1;i++){
		if(skip){
			skip=0;
		}else{
			if(line[i]==n1 && line[i+1]==n2){
				newLine[j]=nextNum;
				--newLen;
				skip=1;
				j++;
			}else{
				newLine[j]=line[i];
				j++;
			}
		}
	}
	newLine[j]=line[i];
	// copy newLine to Line;
	for(int i=0;i<newLen;i++)
		line[i]=newLine[i];
	return newLen;
}
int undoHistory(int history[][4],int lenHistory,int *line,int len){
	// get length of new line from history
	int ints2add=0;
	int k;
	for(int i=0;i<lenHistory;i++)
		ints2add += history[i][0]; // history has 4 columns
	printf("adding %d integers\n",ints2add);
	int *newLine=(int *)calloc(len+ints2add,sizeof(int));
	printf("length of old line is %d\n",len);
	printf("length of new Line is %d\n",len+ints2add);
	for(int i=lenHistory-1;i>=0;i--){
		k=0;
		printf("len is %d\n",len);
		for(int j=0;j<len;j++){
			if(line[j] != history[i][1]){
				newLine[k++] = line[j];
			}else{
				printf("line[%d] = %d\n",j,line[j]);
				newLine[k++] = history[i][2];
				newLine[k++] = history[i][3];
			}
		}
		// copy newLine to Line and update length
		printf("k = %d\n",k);
		for(int h=0;h<k;h++)
			line[h] = newLine[h];
		len=len+history[i][0];
	}
	return len;
}
int main(){
	unsigned char c;
	int line[1000];
	int len=0;
	int tableSize=255;
	int newTableSize;
	int n1=1,n2=1;
	int history[100][4]={{0}};
	int lenHistory=0;
	int max;
	while((c=getchar()) != 255)
   		line[len++]=c;
	printf("Got %d bytes!\n",len);
	for(int i=0;i<len;i++)
		printf("%d ",line[i]);
	printf("\n");
	while(n1>=0 && n2>=0){
		max=1;
		newTableSize=getBytePair(line,len,tableSize,&n1,&n2,&max);
		if(newTableSize > tableSize){
			printf("new table size is %d and n1 is %d and n2 is %d\n",newTableSize,n1,n2);
			tableSize=newTableSize;
			history[lenHistory][0]=max;
			history[lenHistory][1]=tableSize-1;
			history[lenHistory][2]=n1;
			history[lenHistory][3]=n2;
			len=updateLine(line,len,history[lenHistory][1],n1,n2);
			lenHistory++;
			printf("Got %d bytes!\n",len);
			for(int i=0;i<len;i++)
				printf("%d ",line[i]);
			printf("\n");
		}
		else{
			printf("no more duplicate byte pairs\n");
			n1=-1;n2=-1;
		}
	}
	for(int i=0;i<lenHistory;i++)
		printf("%d(%d) --> %d,%d,%d\n",i,history[i][0],history[i][1],history[i][2],history[i][3]);
	len=undoHistory(history,lenHistory,line,len);
	printf("Got %d bytes!\n",len);
	for(int i=0;i<len;i++)
		printf("%d ",line[i]);
	printf("\n");

	return 0;
}
