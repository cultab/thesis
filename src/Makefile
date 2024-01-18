CC = nvcc
NVCC_COMPILE_ONLY_FLAGS=--device-c
CFLAGS = -O1 --std=c++17 -arch=sm_61 -forward-unknown-to-host-compiler
LIBS = -lm
SRC := naive_serial.cpp
OBJ := $(SRC:.cpp=.o)
TARGET := naive_serial

all: $(TARGET)

# dependencies
dataset.o: types.hpp dataset.hpp
types.o: types.hpp
naive_serial.o: dataset.hpp types.hpp


%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

%.o: %.cu
	$(CC) $(CFLAGS) $(NVCC_COMPILE_ONLY_FLAGS) -c $<

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $(TARGET)

mem_test: clean
	$(CC) $(CFLAGS) test.cpp -o test
	-valgrind ./test
	
mem: $(TARGET)
	valgrind ./$(TARGET)

clean:
	rm -f $(OBJ)
	rm -f ./test
	rm -f ./$(TARGET)

run: CFLAGS += -DTRACE=1
run: $(TARGET)
	./$(TARGET)

# nodebug:
# 	cc svm.c -o svm -DTRACE -lm
