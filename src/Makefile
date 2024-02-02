CC := nvcc

SRC := naive_serial.cu
OBJ := $(SRC:.cu=.o)
TARGET := naive_serial

# colorize output of nvcc like gcc
GRC := $(shell command -v grc 2> /dev/null)
ifdef GRC
	CC := grc -e -c conf.gcc nvcc
endif

# relocatable device code
NVCCFLAGS := --device-c

# REF: https://interrupt.memfault.com/blog/best-and-worst-gcc-clang-compiler-flags#-wundef
# REF: https://developers.redhat.com/blog/2018/03/21/compiler-and-linker-flags-gcc

CFLAGS := --std=c++20 -pipe -fmax-errors=1
# dependency generation
CFLAGS += -MMD -MP

CFLAGS += --optimization-info inline --generate-line-info -lineinfo
CFLAGS += -arch=sm_61 -forward-unknown-to-host-compiler --expt-relaxed-constexpr --extended-lambda
# CFLAGS += -arch=sm_61 -forward-unknown-to-host-compiler
CFLAGS += -O1
# CFLAGS += -O2 -free-loop-vectorize
############
# warnings:#
############
CFLAGS += -Wall -Wextra
# -f    WARN: remove -fsanitize=address in benchmarks
# CFLAGS += -fsanitize=address -fno-common
# -ffunction-sections -fdata-sections
# typing
CFLAGS += -Wstrict-null-sentinel -Wdouble-promotion -Wcast-qual -Wlogical-op -Wconversion
# CFLAGS += -Wold-style-cast -Wzero-as-null-pointer-constant 
# control flow
CFLAGS += -Wimplicit-fallthrough=5 -Wnull-dereference -Wduplicated-branches
# OOP
CFLAGS += -Wctor-dtor-privacy -Wnon-virtual-dtor -Wsuggest-final-types -Wsuggest-final-methods -Wsuggest-override
# format
CFLAGS +=  -Wformat=2 -Wformat-overflow -Wformat-security -Wformat-signedness
# useless code
CFLAGS += -Wextra-semi -Wunused-parameter
# CFLAGS += -Wredundant-decls -Wuseless-cast
# other
# CFLAGS += -Wpadded
CFLAGS += -Wshadow

# LDFLAGS := -Wl,--gc-sections,--print-gc-sections
LDLIBS := -lm

all: options $(TARGET)

options:
	@echo build options:
	@echo "CC          = $(CC)"
	@echo "CCFLAGS     ="
	@echo $(CFLAGS) | tr ' ' '\n' | xargs -I {} echo -e '             {}'
	@echo "NVCCFLAGS   = $(NVCCFLAGS)"
	@echo "LDLIBS      = $(LDLIBS)"
	@echo "LDFLAGS     = $(LDFLAGS)"
	@echo "SRC         = $(SRC)"
	@echo "OBJ         = $(OBJ)"

# dependencies
include *.d

# gcc
%.o: %.cpp
	@echo Compiling: $<
	@$(CC) $(CFLAGS) -c $<

# nvcc
%.o: %.cu
	@echo Compiling: $<
	@$(CC) $(CFLAGS) $(NVCCFLAGS) -c $<

# linking
$(TARGET): $(OBJ)
	@echo Linking: $(TARGET)
	@$(CC) $(CFLAGS) $(LDFLAGS) $(LDLIBS) $(OBJ) -o $(TARGET)

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
run: CFLAGS += -g
run: $(TARGET)
	./$(TARGET)

.PHONY: run clean mem mem_test all options *.d
