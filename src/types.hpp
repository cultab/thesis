#include <algorithm>
#ifndef TYPES_HPP
#define TYPES_HPP 1

#include <cstddef>
#include <cstdio>

typedef double f64;
typedef std::size_t size;

namespace types {

const int PRINT_AFTER = 15;
const int PRINT_DIGITS = 1 + PRINT_AFTER + 2;

#define printd(var)                                                                                                    \
    do {                                                                                                               \
        _printd(#var " :\t%*.*lf\n", var);                                                                             \
    } while (0)
#define vec_print(vec)                                                                                                 \
    do {                                                                                                               \
        _vec_print(vec, #vec);                                                                                         \
    } while (0)

const f64 epsilon = 10e-3;

template <typename number> struct vector {
    const size_t cols;
    number *data;

    vector(std::size_t cols) : cols(cols) { this->data = new number[cols]; }
    ~vector() { delete[] this->data; }
    vector(vector &&other) : cols(other.cols) { this->data = std::move(other.data); }
    vector(number *start, number *end) : cols(end - start) { this->data = start; }
    vector &operator=(vector &&other) { return std::move(other.data); }
    number *begin() { return this->data; }
    number *end() { return this->data + cols - 1; }
    number operator[](std::size_t index) { return this->data[index]; }
    auto get(std::size_t index) { return this->data[index]; }
    auto set(std::size_t index, number n) { this->data[index] = n; }
};

template <typename number> struct matrix {
    const size_t rows;
    const size_t cols;
    number *data;

    matrix(std::size_t rows, std::size_t cols) : rows(rows), cols(cols) { this->data = new number[cols * rows]; }
    ~matrix() { delete[] this->data; }
    matrix(matrix &&other) : rows(other.rows), cols(other.cols) { this->data = std::move(other.data); }
    matrix &operator=(matrix &&other) { return std::move(other.data); }
    number *begin() { return this->data; }
    number *end() { return this->data + rows * cols; }
    vector<number> const operator[](std::size_t index) {
        return vector(this->data + index * rows, this->data + index * rows + cols);
    }
    auto inline get(std::size_t row, std::size_t col) { return this->data[row * this->cols + col]; }
};

void _printd(const char *fmt, f64 var);
// void _vec_print(vector v, const char *msg);

#ifdef TRACE
const bool DEBUG = true;
#else
const bool DEBUG = false;
#endif

void inline _printd(const char *fmt, f64 var) {
    if constexpr (DEBUG) {
        std::printf(fmt, PRINT_DIGITS, PRINT_AFTER, var);
    }
}

// void inline _vec_print(vector v, const char *msg) {
//     if constexpr (DEBUG) {
//         std::puts(msg);
//         for (std::size_t j = 0; j < v.cols; j++) {
//             std::printf("%*.*lf ", PRINT_DIGITS, PRINT_AFTER, v.data[j]);
//         }
//         std::puts("");
//     }
// }

} // namespace types
#endif
