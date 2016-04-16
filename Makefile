# This is the program for CSE221 Project
# 2. Memory Operation
#   2.1 RAM access time

CCC         = /usr/bin/
CC          = gcc
OS = $(shell uname)
ifeq ($(OS), Darwin)
    LDFLAGS = -lm
else
    LDFLAGS = -lm -lrt
endif
#CFLAGS      = -Wall -O0
#CFLAGS      = -O3
CFLAGS      = -fopenmp -O3

SRCPATH = ./
OBJPATH = ./OBJ/
INCPATH = ./
LOGPATH = ./LOG/

.PHONY: all clean

all: EXE

$(OBJPATH)%.o: $(SRCPATH)%.c
	$(CC) $(CFLAGS) -c $< -o $@ -I $(INCPATH)

EXE: $(OBJPATH)main.o $(OBJPATH)MLP_3Ls.o $(OBJPATH)my_mat.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJPATH)*.o
	rm -f $(SRCPATH)EXE_MEM
#	rm -f $(LOGPATH)*.*
