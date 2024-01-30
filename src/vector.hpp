#ifndef VECTOR_HPP
#define VECTOR_HPP 1

#include <assert.h>
#include <functional>

#include <cooperative_groups.h>

#include "cuda_helpers.h"
#include "types.hpp"

namespace cg = cooperative_groups;

using cg::grid_group;
using cg::this_grid;
using cg::thread_group;

namespace types {

template <typename T>
struct cuda_vector;

template <typename T>
struct base_vector {
    idx cols;
    T* data;
    bool view = false;

    base_vector()
        : cols(0),
          data(nullptr),
          view(false) {}

    // range constructor
    base_vector(T* start, T* end)
        : cols(end - start),
          data(start),
          view(true) {}

    // sized constructor
    base_vector(idx _cols)
        : cols(_cols),
          data(nullptr),
          view(false) {}

    __host__ __device__ T& operator[](idx i) {
        assert(i < this->cols);
        return this->data[i];
    }

    T* begin() {
        return this->data;
    }
    T* end() {
        return this->data + this->cols - 1;
    }

    // set all elements
    void set(T value) {
        for (idx i = 0; i < this->cols; i++) {
            this->data[i] = value;
        }
    }

    // mutate elements
    // Usage:
    // this->y.mutate([](int x) -> int { return x == 0 ? 1 : -1; });
    void mutate(std::function<T(T)> func) {
        for (idx i = 0; i < this->cols; i++) {
            this->data[i] = func(this->data[i]);
        }
    }
};

template <typename T = number>
struct vector : public base_vector<T> {

    vector() = delete;

    // sized constructor
    vector(idx _cols)
        : base_vector<T>(_cols) {
        this->data = new T[this->cols];
    }

    vector(T* start, T* end)
        : base_vector<T>(start, end) {}

    // move constructor
    vector(vector&& other)
        : base_vector<T>(other.cols) { // TODO: check if base_vector() is called automagically
                                       // DONE: it is
        *this = std::move(other);
    }
    // move assignment
    vector& operator=(vector&& other) {
        // puts("vector<T>::move");
        // printf("addr %p\n", this);
        assert(other.view == false);
        delete[] this->data;
        this->data = other.data;
        this->cols = other.cols;
        other.data = nullptr;
        other.cols = 0;
        return *this;
    }

    // copy constructor
    vector(vector& other)
        : base_vector<T>(other.cols) {
        // puts("vector<T>::copy const");
        *this = other;
    }
    // copy assignment
    vector& operator=(vector& other) {
        // puts("vector<T>::copy assign");
        if (this->cols != other.cols || this->data == nullptr) { // don't delete[n] just to new[n]
            delete[] this->data;
            this->data = new T[other.cols];
            // printf("%p\n", &this->data);
            this->cols = other.cols;
        }
        memcpy(this->data, other.data, sizeof(T) * other.cols);
        return *this;
    }

    // copy conversion constructor
    vector(cuda_vector<T>& other)
        : base_vector<T>(other.cols) {
        *this = other;
    }

    // copy convert
    vector<T>& operator=(cuda_vector<T>& other) {
        if (this->cols != other.cols || this->data == nullptr) {
            delete[] this->data;
            this->data = new T[other.cols];
            this->cols = other.cols;
        }
        cudaErr(cudaMemcpy(this->data, other.data, sizeof(T) * other.cols, cudaMemcpyDeviceToHost));
        return *this;
    }

    ~vector() {
        if (!this->view) {
            delete[] this->data;
        }
    }
    void print(const char* msg);
};

template <>
inline void vector<int>::print(const char* msg) {
    for (idx i = 0; i < this->cols; i++) {
        printf("%s[%zu]: %*d\n", msg, i, PRINT_DIGITS, this->data[i]);
    }
}

template <>
inline void vector<double>::print(const char* msg) {
    for (idx i = 0; i < this->cols; i++) {
        printf("%s[%zu]: %*.*f\n", msg, i, PRINT_DIGITS, PRINT_AFTER, this->data[i]);
    }
}

template <typename T>
struct cuda_vector : public base_vector<T> {
    // sized constructor
    cuda_vector(idx _cols)
        : base_vector<T>(_cols) {
        cudaErr(cudaMalloc(&this->data, sizeof(T) * this->cols));
    }
    cuda_vector(T* start, T* end)
        : base_vector<T>(start, end) {}
    // move constructor
    cuda_vector(cuda_vector&& other) {
        *this = std::move(other);
    }
    // move assignment
    cuda_vector& operator=(cuda_vector&& other) {
        assert(other.view == false);
        cudaErr(cudaFree(this->data));
        this->data = other.data;
        this->cols = other.cols;
        other.data = nullptr;
        other.cols = 0;
        return *this;
    }

    // copy constructor
    cuda_vector(cuda_vector& other)
        : base_vector<T>(other.cols) {
        *this = other;
    }
    // copy assignment
    cuda_vector& operator=(cuda_vector& other) {
        // puts("vector<T>::copy");
        if (this->cols != other.cols || this->data == nullptr) { // don't delete[n] just to new[n]
            cudaErr(cudaFree(this->data));
            cudaErr(cudaMalloc(&this->data, sizeof(T) * other.cols));
            // printf("%p\n", &this->data);
            this->cols = other.cols;
        }
        cudaErr(cudaMemcpy(this->data, other.data, sizeof(T) * other.cols, cudaMemcpyDeviceToDevice));
        return *this;
    }

    // copy conversion constructor
    cuda_vector(vector<T>& other)
        : base_vector<T>(other.cols) {
        *this = other;
    }

    // conversion
    cuda_vector& operator=(vector<T>& other) {
        if (this->cols != other.cols || this->data == nullptr) {
            cudaErr(cudaFree(this->data));
            cudaErr(cudaMalloc(&this->data, sizeof(T) * other.cols));
            this->cols = other.cols;
        }
        cudaErr(cudaMemcpy(this->data, other.data, sizeof(T) * other.cols, cudaMemcpyHostToDevice));
        return *this;
    }

    ~cuda_vector() {
        if (!this->view) {
            cudaErr(cudaFree(this->data));
        }
    }

    // // reduce the array to a single value stored at  TODO: which index?
    // __device__ void reduce(std::function<T(idx)> func) {
    //     unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    //     unsigned int stride = blockDim.x * gridDim.x;
    // }
    //
    // __device__ number min() {
    //     unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    //     unsigned int stride = blockDim.x * gridDim.x;
    //     grid_group grid = this_grid();
    //
    //     number local_min = -types::NUM_MAX;
    //     __shared__ number block_local_min;
    //     __device__ number true_min;
    //
    //     if (tid == 0) {
    //         true_min = -types::NUM_MAX;
    //     }
    //
    //     if (threadIdx.x == 0) {
    //         block_local_min = -types::NUM_MAX;
    //     }
    //
    //     // find local min
    //     for (idx i = tid; i < this->cols; i += stride) {
    //         if (this->data[i] < local_min) {
    //             local_min = this->data[i];
    //         }
    //     }
    //     // find block local min
    //     atomicMin(&block_local_min, local_min);
    //     __syncthreads();
    //
    //     // block leaders find true min
    //     if (threadIdx.x == 0) {
    //         atomicMin(&true_min, block_local_min);
    //     }
    //     grid.sync();
    //     return true_min;
    // }
    //
    // __device__ number max() {
    //     unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    //     unsigned int stride = blockDim.x * gridDim.x;
    //     grid_group grid = this_grid();
    //
    //     number local_max = -types::NUM_MAX;
    //     __shared__ number block_local_max;
    //     __device__ number true_max;
    //
    //     if (tid == 0) {
    //         true_max = -types::NUM_MAX;
    //     }
    //
    //     if (threadIdx.x == 0) {
    //         block_local_max = -types::NUM_MAX;
    //     }
    //
    //     // find local max
    //     for (idx i = tid; i < this->cols; i += stride) {
    //         if (this->data[i] < local_max) {
    //             local_max = this->data[i];
    //         }
    //     }
    //     // find block local max
    //     atomicMax(&block_local_max, local_max);
    //     __syncthreads();
    //
    //     // block leaders find true max
    //     if (threadIdx.x == 0) {
    //         atomicMax(&true_max, block_local_max);
    //     }
    //     grid.sync();
    //     return true_max;
    // }

    __device__ void mutate(std::function<__device__ T(idx)> func) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int stride = blockDim.x * gridDim.x;
        grid_group grid = this_grid();

        for (idx i = tid; i < this->cols; i += stride) {
            this->data[i] = func(i);
        }
        grid.sync();
    }

    __device__ void set(T value) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int stride = blockDim.x * gridDim.x;
        grid_group grid = this_grid();
        for (idx i = tid; i < this->cols; i += stride) {
            this->data[i] = value;
        }
        grid.sync();
    }
};

typedef number (*Kernel)(base_vector<number>, base_vector<number>);

// using Kernel = std::function<number(vector<number>, vector<number>)>;

template <typename T>
void _printd(vector<T>& vec, const char* msg) {
    vec.print(msg);
}

} // namespace types
#endif
