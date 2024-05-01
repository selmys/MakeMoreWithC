#include <stdio.h>
int main(){
	float total=0.0;
	float n;
	for(int i=0;i<32;i++){
		scanf("%e",&n);
		total+=n;
		printf("n = %.6e and total is %.6e\n",n,total);
	}
	return 0;
}
