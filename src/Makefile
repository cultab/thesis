CC = g++
CFLAGS = -O1 --std=c++17 -DTRACE
LIBS = -lm
SRC := naive_serial.cpp
OBJ := $(SRC:.cpp=.o)
TARGET := svm

all: $(TARGET)

# dependencies
dataset.o: types.hpp dataset.hpp
types.o: types.hpp
naive_serial.o: dataset.hpp types.hpp


# %.o: %.cpp
# 	$(CC) $(CFLAGS) -c $<

$(TARGET): CFLAGS += -O3
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $(TARGET)

mem_test: clean
	$(CC) $(CFLAGS) test.cpp -o test
	-valgrind ./test
	
mem: $(TARGET)
	valgrind ./svm

clean:
	rm -f $(OBJ)
	rm -f ./test
	rm -f ./$(TARGET)

run: clean $(TARGET)
	./svm

# nodebug:
# 	cc svm.c -o svm -DTRACE -lm
