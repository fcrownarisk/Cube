# Makefile for Cube4096 System

CC = gcc
CFLAGS = -Wall -Wextra -O2 -lm
TARGET = cube4096
HEADERS = cube4096_utils.h
SOURCES = cube4096_main.c cube4096_utils.c

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) cube4096_main.c cube4096_utils.c -lm

debug: $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -g -o $(TARGET)_debug cube4096_main.c cube4096_utils.c -lm

release: $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -O3 -o $(TARGET)_release cube4096_main.c cube4096_utils.c -lm

clean:
	rm -f $(TARGET) $(TARGET)_debug $(TARGET)_release *.o *.dat *.obj

run: $(TARGET)
	./$(TARGET)

test: debug
	./$(TARGET)_debug

.PHONY: all clean run test debug release