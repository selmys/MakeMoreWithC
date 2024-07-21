#include <stdio.h>
int main(){
	int i;
	while((i=getchar())!= EOF)
		printf("%d ",i);
	printf("\n");
	return 0;
}
