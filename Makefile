CC = gcc
CFLAGS = -Wall -g -lm
BINS = main.o data_io.o layer.o network.o convolution.o util.o

all: $(BINS)
	$(CC) $(CFLAGS) $(BINS) -o main

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $^

clean:
	rm *.o *.gch main