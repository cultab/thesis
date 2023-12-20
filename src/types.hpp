#include <algorithm>
#include <functional>
#ifndef TYPES_HPP
#define TYPES_HPP 1

#include <cstddef>
#include <cstdio>

namespace types {

using f64 = double;
using size = std::size_t;
const int PRINT_AFTER = 25;
const int PRINT_DIGITS = 1 + PRINT_AFTER + 2;

#define printd(var)                                                                                                    \
    do {                                                                                                               \
        types::_printd(#var " :\t%*.*lf\n", var);                                                                      \
    } while (0)
#define vec_print(vec)                                                                                                 \
    do {                                                                                                               \
        types::_vec_print(vec, #vec);                                                                                  \
    } while (0)

const f64 epsilon = 1e-3;

template <typename number = f64, bool owns_memory = true> struct vector {
    const size_t cols;
    number* data;

    // auto get(std::size_t index) {
    //     return this->data[index];
    // }

    // range constructor
    vector(number* start, number* end) : cols(end - start) {
        this->data = start;
    }
    // sized constructor
    vector(std::size_t cols) : cols(cols) {
        this->data = new number[cols];
    }
    // move constructor
    vector(vector&& other) : cols(other.cols) {
        this->data = std::move(other.data);
    }
    // explicit conversion constructor
    // vector(vector<number, false>&& other) : cols(other.cols) {
    //     this->data = std::move(other.data);
    // }

    // move assignment
    vector& operator=(vector&& other) {
        delete[] this->data;
        this->data = other.data;
        this->cols = other.cols;
        return *this;
    }

    // delete copy constructor and assignment
    vector(const vector& other) = delete;
    vector& operator=(vector& other) = delete;

    ~vector() {
        if constexpr (owns_memory) {
            delete[] this->data;
        }
    }

    // set all elements
    void set(number value) {
        /* clang-format off  */
        #pragma omp parallel for /* clang-format on  */
        for (size i = 0; i < this->cols; i++) {
            this->data[i] = value;
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

    // mutate elements
    void mutate(std::function<number(number)> func) {
        /* clang-format off  */
        #pragma omp parallel for /* clang-format on  */
        for (size i = 0; i < this->cols; i++) {
            this->data[i] = func(this->data[i]);
        }
    }
};

template <typename number> struct matrix {
    const size_t rows;
    const size_t cols;
    number* data;

    // auto inline get(std::size_t row, std::size_t col) {
    //     return this->data[row * this->cols + col];
    // }

    // sized constructor
    matrix(std::size_t rows, std::size_t cols) : rows(rows), cols(cols) {
        this->data = new number[cols * rows];
    }
    // move constructor
    matrix(matrix&& other) : rows(other.rows), cols(other.cols) {
        this->data = std::move(other.data);
    }

    // move assignment
    matrix& operator=(matrix&& other) {
        delete[] this->data;
        this->data = other.data;
        this->cols = other.cols;
        this->rows = other.rows;
        return *this;
    }

    ~matrix() {
        delete[] this->data;
    }

    number* begin() {
        return this->data;
    }
    number* end() {
        return this->data + rows * cols;
    }
    // returns a vector which does not deallocate it's data, since it's owned by this matrix
    vector<number, false> operator[](std::size_t index) {
        return vector<number, false>(&(this->data[index * cols]), &(this->data[index * cols + cols]));
    }
    vector<number, false> operator[](std::size_t index) const {
        return vector<number, false>(&(this->data[index * cols]), &(this->data[index * cols + cols]));
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

void inline _vec_print(const matrix<f64>& v, const char* msg) {
    if constexpr (DEBUG) {
        std::puts(msg);
        for (std::size_t i = 0; i < v.rows; i++) {
            for (std::size_t j = 0; j < v.cols; j++) {
                std::printf("%*.*f ", PRINT_DIGITS, PRINT_AFTER, v[i][j]);
            }
            std::puts("");
        }
    }
}

} // namespace types
#endif
