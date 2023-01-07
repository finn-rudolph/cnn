CC = clang
CFLAGS = -Wall -O3 -march=native -DNDEBUG
BINS = main.o image_data.o layer.o network.o convolution.o

all: $(BINS) main.o
	$(CC) $(CFLAGS) $(BINS) -lm -o main

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $^

clean:
	rm *.o *.gch main