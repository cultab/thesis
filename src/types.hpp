#include <algorithm>
#include <cfloat>
#include <functional>
#ifndef TYPES_HPP
#define TYPES_HPP 1

#include <cstddef>
#include <cstdio>

namespace types {

using f64 = double;
using label = int;
using index = size_t;
const int PRINT_AFTER = 15;
const int PRINT_DIGITS = 1 + PRINT_AFTER + 2;
const f64 f64_max = DBL_MAX;

#define printd(var)                                                                                                    \
    do {                                                                                                               \
        types::_printd(#var " :\t%*.*lf\n", var);                                                                      \
    } while (0)
#define vec_print(vec)                                                                                                 \
    do {                                                                                                               \
        types::_vec_print(vec, #vec);                                                                                  \
    } while (0)

#define printc(cond)                                                                                                   \
    do {                                                                                                               \
        types::_printc(#cond ": \t%s\n", cond ? "true" : "false");                                                     \
    } while (0)

const f64 epsilon = DBL_EPSILON;
// const f64 epsilon = 0.001;

template <typename number = f64, bool owns_memory = true> struct vector {
    const index cols;
    number* data;

    // auto get(index index) {
    //     return this->data[index];
    // }

    // range constructor
    vector(number* start, number* end) : cols(end - start) {
        this->data = start;
    }
    // sized constructor
    vector(index cols) : cols(cols) {
        this->data = new number[cols];
    }
    // move constructor
    vector(vector&& other) : cols(other.cols) {
        this->data = std::move(other.data);
        other.data = nullptr;
    }
    // explicit conversion constructor
    // vector(vector<number, false>&& other) : cols(other.cols) {
    //     this->data = std::move(other.data);
    // }
    vector copy() {
        vector copy(this->cols);
        memcpy(copy.data, this->data, this->cols * sizeof(number));
        return copy;
    }

    // move assignment
    vector& operator=(vector&& other) {
        delete[] this->data;
        this->data = other.data;
        this->cols = other.cols;
        other.data = nullptr;
        return *this;
    }

    // delete copy constructor and assignment
    vector(const vector& other) = delete;
    vector& operator=(vector& other) = delete;

    ~vector() {
        if constexpr (owns_memory) {
            // printf("~vector: %p\n", this);
            delete[] this->data;
        }
    }

    // set all elements
    void set(number value) {
/* clang-format off  */
#pragma omp parallel for /* clang-format on  */
        for (index i = 0; i < this->cols; i++) {
            this->data[i] = value;
        }
    }

    number& operator[](index index) {
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
        for (index i = 0; i < this->cols; i++) {
            this->data[i] = func(this->data[i]);
        }
    }
};

// TODO: types::sparse_vector aka a vec of the indexes of non zero values and a vec of values
//
template <typename number = f64> struct sparse_vector : vector<number> {
    index* indexes;
    index count;

    // sized constructor
    sparse_vector(index cols) : vector<number>(cols) {
        this->indexes = new number[cols];
    }
    // move constructor
    sparse_vector(vector<number>&& other) : vector<number>(other) {
        this->indexes = std::move(other.data);
        other.indexes = nullptr;
    }
    // explicit conversion constructor
    // vector(vector<number, false>&& other) : cols(other.cols) {
    //     this->data = std::move(other.data);
    // }

    // move assignment
    sparse_vector& operator=(vector<number>&& other) {
        vector<number>::operator=(other);
        delete[] this->indexes;
        this->data = other.indexes;
        this->cols = other.cols;
        other.indexes = nullptr;
        return *this;
    }

    // delete copy constructor and assignment
    sparse_vector(const vector<number>& other) = delete;
    sparse_vector& operator=(vector<number>& other) = delete;

    bool has(index index) {
        for (::types::index i = 0; i < this->count; i++) {
            ::types::index index = indexes[i];
            if (index == i) {
                return true;
            }
        }
        return false;
    }

    number& operator[](index index) {
        // find index in indexes
        for (::types::index i = 0; i < this->cols; i++) {
            ::types::index index = indexes[i];
            if (index == i) {
                // if found return data at i
                return this->data[i];
            }
        }
        // else if not found
        return 0;
    }
};

template <typename number> struct matrix {
    const index rows;
    const index cols;
    number* data;

    // auto inline get(index row, index col) {
    //     return this->data[row * this->cols + col];
    // }

    // sized constructor
    matrix(index rows, index cols) : rows(rows), cols(cols) {
        this->data = new number[cols * rows];
    }
    // move constructor
    matrix(matrix&& other) : rows(other.rows), cols(other.cols) {
        this->data = std::move(other.data);
        other.data = nullptr;
    }

    // move assignment
    matrix& operator=(matrix&& other) {
        delete[] this->data;
        this->data = other.data;
        this->cols = other.cols;
        this->rows = other.rows;
        other.data = nullptr;
        return *this;
    }

    ~matrix() {
        // printf("~matrix: %p\n", this);
        delete[] this->data;
    }

    number* begin() {
        return this->data;
    }
    number* end() {
        return this->data + rows * cols;
    }
    // returns a vector which does not deallocate it's data, since it's owned by this matrix
    vector<number, false> operator[](index index) {
        return vector<number, false>(&(this->data[index * cols]), &(this->data[index * cols + cols]));
    }
    vector<number, false> operator[](index index) const {
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
        printf(fmt, PRINT_DIGITS, PRINT_AFTER, var);
    }
}

void inline _printc(const char* fmt, const char* b) {
    if constexpr (DEBUG) {
        printf(fmt, b);
    }
}

void inline _vec_print(const vector<int>& v, const char* msg) {
    if constexpr (DEBUG) {
        puts(msg);
        for (index j = 0; j < v.cols; j++) {
            printf("%*.*d ", PRINT_DIGITS, PRINT_AFTER, v.data[j]);
        }
        puts("");
    }
}

void inline _vec_print(const vector<f64>& v, const char* msg) {
    if constexpr (DEBUG) {
        puts(msg);
        for (index j = 0; j < v.cols; j++) {
            printf("%*.*lf", PRINT_DIGITS, PRINT_AFTER, v.data[j]);
            if (j % 4 == 0) {
                puts("");
            }
        }
        puts("");
    }
}

void inline _vec_print(const matrix<f64>& v, const char* msg) {
    if constexpr (DEBUG) {
        puts(msg);
        for (index i = 0; i < v.rows; i++) {
            for (index j = 0; j < v.cols; j++) {
                printf("%*.*f ", PRINT_DIGITS, PRINT_AFTER, v[i][j]);
            }
            puts("");
        }
    }
}

typedef f64 (*Kernel)(vector<f64, false>, vector<f64, false>);
} // namespace types
#endif
