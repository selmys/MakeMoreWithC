#include <stdio.h>
int main(){
	int x[84];
	int i=0;
	while((x[i]=getchar())!= EOF)
		i++;
	for(i=0;i<84;i++)
		putchar(x[i]);
	return 0;
}
