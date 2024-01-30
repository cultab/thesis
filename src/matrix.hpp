#ifndef MATRIX_HPP
#define MATRIX_HPP 1

#include "types.hpp"
#include "vector.hpp"

namespace types {

struct base_matrix {
    idx rows;
    idx cols;
    number* data;
    base_matrix()
        : rows(0),
          cols(0),
          data(nullptr) {}
    base_matrix(idx _rows, idx _cols)
        : rows(_rows),
          cols(_cols),
          data(nullptr) {}

    number* begin() {
        return this->data;
    }

    number* end() {
        return this->data + rows * cols;
    }
};

struct matrix : base_matrix {

    // sized constructor
    matrix(idx _rows, idx _cols)
        : base_matrix(_rows, _cols) {
        this->data = new number[_cols * _rows];
    }
    // move constructor
    matrix(matrix&& other)
        : base_matrix(other.rows, other.cols) {
        *this = std::move(other);
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

    matrix(matrix& other)
        : base_matrix(other.rows, other.cols) {
        *this = other;
    }

    matrix& operator=(matrix& other) {
        if (this->cols * this->rows != other.cols * other.rows) {
            delete[] this->data;
            this->data = new number[cols * rows];
            this->cols = other.cols;
            this->rows = other.cols;
        }
        memcpy(this->data, other.data, rows * cols * sizeof(number));
        return *this;
    }

    ~matrix() {
        // printf("~matrix: %p\n", this);
        delete[] this->data;
    }

    // returns a vector which does not deallocate it's data, since it's owned by this matrix
    vector<number> operator[](idx index) {
        return vector<number>(&(this->data[index * cols]), &(this->data[index * cols + cols]));
    }
    vector<number> operator[](idx index) const {
        return vector<number>(&(this->data[index * cols]), &(this->data[index * cols + cols]));
    }

    void print() {
        for (idx i = 0; i < this->rows; i++) {
            for (idx j = 0; j < this->cols; j++) {
                printf("%*.*f ", PRINT_DIGITS, PRINT_AFTER, this->data[i * this->cols + j]);
            }
            puts("");
        }
    }
};

void inline _printd(matrix& mat, const char* msg) {
    puts(msg);
    mat.print();
}

struct cuda_matrix : base_matrix {

    // sized constructor
    cuda_matrix(idx _rows, idx _cols)
        : base_matrix(_rows, _cols) {
        cudaErr(cudaMalloc(&this->data, sizeof(number) * _cols * _rows));
    }
    // move constructor
    cuda_matrix(matrix&& other)
        : base_matrix(other.rows, other.cols) {
        *this = std::move(other);
    }

    // move assignment
    cuda_matrix& operator=(matrix&& other) {
        cudaErr(cudaFree(this->data));
        this->data = other.data;
        this->cols = other.cols;
        this->rows = other.rows;
        other.data = nullptr;
        return *this;
    }

    cuda_matrix(matrix& other)
        : base_matrix(other.rows, other.cols) {
        *this = other;
    }

    cuda_matrix& operator=(matrix& other) {
        if (this->cols * this->rows != other.cols * other.rows) {
            cudaErr(cudaFree(this->data));
            cudaErr(cudaMalloc(&this->data, sizeof(number) * other.cols * other.rows));
            this->cols = other.cols;
            this->rows = other.cols;
        }
        memcpy(this->data, other.data, rows * cols * sizeof(number));
        return *this;
    }

    ~cuda_matrix() {
        // printf("~matrix: %p\n", this);
        cudaErr(cudaFree(this->data));
    }

    // returns a vector which does not deallocate it's data, since it's owned by this cuda_matrix
    __device__ cuda_vector<number> operator[](idx index) {
        return cuda_vector<number>(&(this->data[index * cols]), &(this->data[index * cols + cols]));
    }
    __device__ cuda_vector<number> operator[](idx index) const {
        return cuda_vector<number>(&(this->data[index * cols]), &(this->data[index * cols + cols]));
    }
};

} // namespace types
#endif
