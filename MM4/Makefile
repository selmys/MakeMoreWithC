CC = gcc 
CFLAGS = -Wall -Wextra -O 
main: main.o helpers.o printers.o forepass.o backpass.o testing.o normalize.o matrix.o 
	$(CC) -o main main.o helpers.o printers.o forepass.o backpass.o testing.o normalize.o matrix.o -lm -lgsl
main.o: main.c helpers.h printers.h forepass.h backpass.h normalize.h matrix.h testing.h params.h
	$(CC) -c $(CFLAGS) main.c 
helpers.o: helpers.c helpers.h
	$(CC) -c $(CFLAGS) helpers.c 
printers.o: printers.c printers.h helpers.h params.h
	$(CC) -c $(CFLAGS) printers.c 
forepass.o: forepass.c forepass.h helpers.h params.h
	$(CC) -c $(CFLAGS) forepass.c 
backpass.o: backpass.c backpass.h matrix.h params.h
	$(CC) -c $(CFLAGS) backpass.c 
testing.o: testing.c testing.h normalize.h matrix.h forepass.h backpass.h params.h
	$(CC) -c $(CFLAGS) testing.c 
normalize.o: normalize.c normalize.h params.h
	$(CC) -c $(CFLAGS) normalize.c
matrix.o: matrix.c 
	$(CC) -c $(CFLAGS) matrix.c
clean:
	rm *.o main
