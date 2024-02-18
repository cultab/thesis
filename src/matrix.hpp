#ifndef MATRIX_HPP
#define MATRIX_HPP 1

#include "types.hpp"
#include "vector.hpp"

namespace types {

template <typename T>
struct cuda_matrix;

template <typename T>
struct base_matrix {
    idx rows;
    idx cols;
    T* data;
    base_matrix()
        : rows(0),
          cols(0),
          data(nullptr) {}
    base_matrix(idx _rows, idx _cols)
        : rows(_rows),
          cols(_cols),
          data(nullptr) {}

    T* begin() {
        return this->data;
    }

    T* end() {
        return this->data + rows * cols;
    }

    auto shape() {
        return std::make_tuple(rows, cols);
    }
};

template <typename T>
struct matrix : public base_matrix<T> {

    // sized constructor
    matrix(idx _rows, idx _cols)
        : base_matrix<T>(_rows, _cols) {
        this->data = new T[_cols * _rows];
    }
    // move constructor
    matrix(matrix&& other)
        : base_matrix<T>(other.rows, other.cols) {
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
        : base_matrix<T>(other.rows, other.cols) {
        *this = other;
    }

    matrix& operator=(matrix& other) {
        if (this->cols * this->rows != other.cols * other.rows) {
            delete[] this->data;
            this->data = new T[other.cols * other.rows];
            this->cols = other.cols;
            this->rows = other.cols;
        }
        memcpy(this->data, other.data, other.rows * other.cols * sizeof(T));
        return *this;
    }

    matrix(cuda_matrix<T>& other)
        : base_matrix<T>(other.rows, other.cols) {
        *this = other;
    }

    matrix& operator=(cuda_matrix<T>& other) {
        if (this->cols * this->rows != other.cols * other.rows || this->data == nullptr) {
            free(this->data);
            this->data = new T[other.cols * other.rows];
            this->cols = other.cols;
            this->rows = other.rows;
        }
        cudaErr(cudaMemcpy(this->data, other.data, sizeof(T) * other.cols * other.rows, cudaMemcpyDeviceToHost));
        return *this;
    }

    ~matrix() {
        // printf("~matrix: %p\n", this);
        delete[] this->data;
    }

    // returns a vector which does not deallocate it's data, since it's owned by this matrix
    vector<T> operator[](idx index) {
        assert(index < this->rows);
        return vector<T>(&(this->data[index * this->cols]), &(this->data[index * this->cols + this->cols]));
    }
    vector<T> operator[](idx index) const {
        assert(index < this->rows);
        return vector<T>(&(this->data[index * this->cols]), &(this->data[index * this->cols + this->cols]));
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

template <typename T>
void inline _printd(matrix<T>& mat, const char* msg) {
    puts(msg);
    mat.print();
}

template <typename T>
struct cuda_matrix : base_matrix<T> {

    // sized constructor
    cuda_matrix(idx _rows, idx _cols)
        : base_matrix<T>(_rows, _cols) {
        cudaErr(cudaMalloc(&this->data, sizeof(T) * _cols * _rows));
    }
    // move constructor
    cuda_matrix(matrix<T>&& other)
        : base_matrix<T>(other.rows, other.cols) {
        *this = std::move(other);
    }

    // move assignment
    cuda_matrix& operator=(matrix<T>&& other) {
        cudaErr(cudaFree(this->data));
        this->data = other.data;
        this->cols = other.cols;
        this->rows = other.rows;
        other.data = nullptr;
        return *this;
    }

    cuda_matrix(matrix<T>& other)
        : base_matrix<T>(other.rows, other.cols) {
        *this = other;
    }

    cuda_matrix& operator=(matrix<T>& other) {
        if (this->cols * this->rows != other.cols * other.rows || this->data == nullptr) {
            cudaErr(cudaFree(this->data));
            cudaErr(cudaMalloc(&this->data, sizeof(T) * other.cols * other.rows));
            this->cols = other.cols;
            this->rows = other.rows;
        }
        cudaErr(cudaMemcpy(this->data, other.data, sizeof(T) * other.cols * other.rows, cudaMemcpyHostToDevice));
        return *this;
    }

    ~cuda_matrix() {
        // printf("~matrix: %p\n", this);
        cudaErr(cudaFree(this->data));
    }

    // returns a vector which does not deallocate it's data, since it's owned by this cuda_matrix
    __device__ cuda_vector<T> operator[](idx index) {
        return cuda_vector<T>(&(this->data[index * this->cols]), &(this->data[index * this->cols + this->cols]));
    }
    __device__ cuda_vector<T> operator[](idx index) const {
        return cuda_vector<T>(&(this->data[index * this->cols]), &(this->data[index * this->cols + this->cols]));
    }
};

} // namespace types
#endif
