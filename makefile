SYSTEMC_HOME ?= /usr/local/systemc

CXX      = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2 -g
CXXFLAGS += -I$(SYSTEMC_HOME)/include -Isrc -Isrc/core
LDFLAGS  = -L$(SYSTEMC_HOME)/lib -L. -Wl,-rpath,$(SYSTEMC_HOME)/lib -pthread -lsystemc -lm -lpthread

SRCDIR = src
TARGET = out

.PHONY: all clean run

all: $(TARGET)

 $(TARGET): main.cpp
    $(CXX) $(CXXFLAGS) main.cpp -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
    ./$(TARGET)

clean:
    rm -f $(TARGET)