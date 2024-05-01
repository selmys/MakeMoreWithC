reset
set autoscale fix
set title 'one hot values'
set nokey
set palette defined (0 'gray', 1 'dark-red')
set datafile separator ';'
set yrange reverse
#plot 'onehot.txt' matrix rowheaders columnheaders u 1:2:3 with image
unset border
set tics scale 0
plot 'onehot.txt' matrix rowheaders columnheaders u 1:2:(sprintf('%g',$3)) with labels font ',8'
pause mouse
