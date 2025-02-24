# Compiler and flags
CC = gcc
CFLAGS = -Wall -O0
LDFLAGS = -lm -lpng16

# Targets
all: process

main_process.o: main_process.c
	$(CC) $(CFLAGS) -c main_process.c

png_util.o: png_util.c
	$(CC) $(CFLAGS) -c png_util.c

process: main_process.o png_util.o
	$(CC) -o process main_process.o png_util.o $(LDFLAGS)

test: process
	./process ./images/cube.png test.png

benchmark: process
	@echo "Running benchmarks..."
	@echo "======================"
	@for image in cube.png sparty.png earth.png MSUStadium.png; do \
		echo "Processing ./images/$$image..."; \
		for trial in 1 2 3 4 5; do \
			echo "Trial $$trial:"; \
			./process ./images/$$image test_$$image; \
		done; \
		echo ""; \
	done

clean:
	rm -f *.o process test_*.png
