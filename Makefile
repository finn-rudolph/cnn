CC = clang
CFLAGS = -Wall -g -lm
BINS = main.o image_data.o layer.o network.o convolution.o

all: $(BINS)
	$(CC) $(CFLAGS) $(BINS) -o main

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $^

clean:
	rm *.o *.gch main