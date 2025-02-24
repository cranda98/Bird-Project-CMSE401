CFLAGS = -Wall -O3 -fopenmp

all: process

main_process.o: main_process.c
	gcc $(CFLAGS) -c main_process.c

png_util.o: png_util.c
	gcc $(CFLAGS) -c png_util.c

process: main_process.o png_util.o
	gcc -o process main_process.o png_util.o -lm -lpng16 -fopenmp

test: process
	./process ./images/cube.png test.png

clean:
	rm -f *.o process
