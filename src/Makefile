CC = g++
CFLAGS = -O1
LIBS = -lm
SRC := $(wildcard *.cpp)
OBJ := $(SRC:.cpp=.o)
TARGET := svm

all: $(TARGET)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

$(TARGET): CFLAGS += -O3
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $(TARGET)

# nodebug:
# 	cc svm.c -o svm -DTRACE -lm
