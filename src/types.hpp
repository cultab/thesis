#include <cstring>
#ifndef TYPES_HPP
#define TYPES_HPP 1

#include <cfloat>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>

#ifdef TRACE
const bool DEBUG = true;
#else
const bool DEBUG = false;
#endif

namespace types {

using math_t = double;
const math_t MATH_T_MAX = DBL_MAX;
using label = int;
using idx = size_t;
const int PRINT_AFTER = 24;
const int PRINT_DIGITS = 4 + PRINT_AFTER + 2;

#define printd(var)                                                                                                    \
    do {                                                                                                               \
        types::_printd(var, #var);                                                                                     \
    } while (0)

#define printc(cond)                                                                                                    \
    do {                                                                                                               \
        types::_printc(cond, #cond);                                                                                     \
    } while (0)

const math_t epsilon = DBL_EPSILON;
// const f64 epsilon = 0.001;

void _printd(const char* fmt, bool cond);
void _printd(const char* fmt, math_t var);

void inline _printd(math_t var, const char* msg) {
    if constexpr (DEBUG) {
        puts(msg);
        printf(":\t%*.*lf\n", PRINT_DIGITS, PRINT_AFTER, var);
    }
}

void inline _printc(bool cond, const char* msg) {
    if constexpr (DEBUG) {
        puts(msg);
        printf(": \t%s\n", cond ? "true" : "false");
    }
}

} // namespace types
#endif
