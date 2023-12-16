#include <algorithm>
#ifndef TYPES_HPP
#define TYPES_HPP 1

#include <cstddef>
#include <cstdio>

namespace types {

using f64 = double;
using size = std::size_t;
const int PRINT_AFTER = 15;
const int PRINT_DIGITS = 1 + PRINT_AFTER + 2;

#define printd(var)                                                                                                    \
    do {                                                                                                               \
        types::_printd(#var " :\t%*.*lf\n", var);                                                                      \
    } while (0)
#define vec_print(vec)                                                                                                 \
    do {                                                                                                               \
        types::_vec_print(vec, #vec);                                                                                  \
    } while (0)

const f64 epsilon = 10e-3;

template <typename number = f64, bool parallel = false, bool is_view = false> struct vector {
    const size_t cols;
    number* data;

    auto get(std::size_t index) {
        return this->data[index];
    }
    vector& operator=(vector&& other) {
        delete[] this->data;
        this->data = other.data;
        this->cols = other.cols;
        return *this;
    }
    vector& operator=(vector& other) = delete;
    vector(number* start, number* end) : cols(end - start) {
        this->data = start;
    }
    vector(std::size_t cols) : cols(cols) {
        this->data = new number[cols];
    }
    vector(vector&& other) : cols(other.cols) {
        this->data = std::move(other.data);
    }

    vector(const vector& other) = delete;

    void set(number value) {
        if constexpr (parallel) {
#pragma omp parallel for
            for (size i = 0; i < this->cols; i++) {
                this->data[i] = value;
            }
        } else {
            for (size i = 0; i < this->cols; i++) {
                this->data[i] = value;
            }
        }
    }
    number& operator[](std::size_t index) {
        return this->data[index];
    }
    number* begin() {
        return this->data;
    }
    number* end() {
        return this->data + cols - 1;
    }
    ~vector() {
        if constexpr (!is_view) {
            delete[] this->data;
        }
    }
};

template <typename number, bool parallel = false> struct matrix {
    const size_t rows;
    const size_t cols;
    number* data;

    auto inline get(std::size_t row, std::size_t col) {
        return this->data[row * this->cols + col];
    }
    matrix& operator=(matrix&& other) {
        delete[] this->data;
        this->data = other.data;
        this->cols = other.cols;
        this->rows = other.rows;
        return *this;
    }
    matrix(matrix&& other) : rows(other.rows), cols(other.cols) {
        this->data = std::move(other.data);
    }
    matrix(std::size_t rows, std::size_t cols) : rows(rows), cols(cols) {
        this->data = new number[cols * rows];
    }
    number* begin() {
        return this->data;
    }
    number* end() {
        return this->data + rows * cols;
    }
    vector<number, parallel, true> operator[](std::size_t index) {
        return vector<number, parallel, true>(&(this->data[index * cols]), &(this->data[index * cols + cols]));
    }
    ~matrix() {
        delete[] this->data;
    }
};

void _printd(const char* fmt, f64 var);
// void _vec_print(vector v, const char *msg);

#ifdef TRACE
const bool DEBUG = true;
#else
const bool DEBUG = false;
#endif

void inline _printd(const char* fmt, f64 var) {
    if constexpr (DEBUG) {
        std::printf(fmt, PRINT_DIGITS, PRINT_AFTER, var);
    }
}

void inline _vec_print(const vector<int>& v, const char* msg) {
    if constexpr (DEBUG) {
        std::puts(msg);
        for (std::size_t j = 0; j < v.cols; j++) {
            std::printf("%*.*d ", PRINT_DIGITS, PRINT_AFTER, v.data[j]);
        }
        std::puts("");
    }
}

void inline _vec_print(const vector<f64>& v, const char* msg) {
    if constexpr (DEBUG) {
        std::puts(msg);
        for (std::size_t j = 0; j < v.cols; j++) {
            std::printf("%*.*lf ", PRINT_DIGITS, PRINT_AFTER, v.data[j]);
        }
        std::puts("");
    }
}

} // namespace types
#endif
