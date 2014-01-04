HOME = /home/ff/cs61c
UNAME = $(shell uname)

# running on hive machines
ifeq ($(UNAME),Linux)
CC = gcc -std=gnu99 -O3
GOTO = $(HOME)/bin/GotoBLAS2_Linux
GOTOLIB = $(GOTO)/libgoto2_nehalemp-r1.13.a
endif

# running on 200 SD machines
ifeq ($(UNAME),Darwin)
CC = gcc -std=gnu99 -O3
GOTO = $(HOME)/bin/GotoBLAS2
GOTOLIB = $(GOTO)/libgoto2_nehalemp-r1.13.a
endif

INCLUDES = -I$(GOTO)
OMP = -fopenmp
LIBS = -lpthread  
# a pretty good flag selection for this machine...
CFLAGS = -msse4 -fopenmp -O3 -pipe -fno-omit-frame-pointer

all:	bench-naive bench-part1 bench-part2

# triple nested loop implementation
bench-naive: benchmark.o naive.o ref.o
	$(CC) -o $@ $(LIBS) benchmark.o naive.o ref.o $(GOTOLIB)

# your implementation for part 1
bench-part1: benchmark.o part1.o ref.o
	$(CC) -o $@ $(LIBS) benchmark.o part1.o ref.o $(GOTOLIB)
# your implementation for part 2
bench-part2: benchmark.o part2.o ref.o
	$(CC) -o $@ $(LIBS) $(OMP) benchmark.o part2.o ref.o $(GOTOLIB)

%.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDES) $<

clean:
	rm -f *~ bench-naive bench-part1 bench-part2 *.o
