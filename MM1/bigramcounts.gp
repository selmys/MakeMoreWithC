reset
set autoscale fix
set title 'Bigram Counts'
set nokey
set palette defined (0 'white', 1 'dark-green')
set datafile separator ';'
set yrange reverse
plot 'bigramcounts.txt' matrix rowheaders columnheaders u 1:2:3 with image,\
                     '' matrix rowheaders columnheaders using 1:2:(sprintf('%g',$3)) with labels font ',8'
pause mouse
