CC = gcc
CFLAGS = -Wall -g
BINS = main.o file_io.o layer.o network.o convolution.o util.o

all: $(BINS)
	$(CC) $(CFLAGS) $(BINS) -o main

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $^

clean:
	rm *.o *.gch main