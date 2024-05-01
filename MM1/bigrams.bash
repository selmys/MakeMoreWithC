#!/usr/bin/bash
names=$(cat names.txt | tr '\n' '.')
names='.'$names
l=$(echo ${#names})
((l--))
for((i=0;i<$l;i++)) 
do 
	echo -n ${names:$i:1} 
	((j=i+1)) 
	echo " "${names:$j:1} 
done | sort | uniq -c
