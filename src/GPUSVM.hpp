#ifndef SVM_HPP
#define SVM_HPP 1

#include <cmath>
#include <iostream>

#include <cooperative_groups.h>

#include "SVM_common.hpp"
#include "dataset.hpp"
#include "matrix.hpp"
#include "types.hpp"
#include "vector.hpp"

#define max(a, b) a > b ? a : b
#define min(a, b) a < b ? a : b

namespace cg = cooperative_groups;
using cg::grid_group;
using cg::this_grid;
using cg::this_thread_block;
using cg::thread_block;
using types::cuda_matrix;
using types::cuda_vector;
using types::idx;
using types::Kernel;
using types::label;
using types::matrix;
using types::number;
using types::vector;

namespace SVM {

__device__ static float atomicMax(double* address, double val) {
    unsigned long long* address_as_i = (unsigned long long*)address;
    unsigned long long old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed, __double_as_longlong(::fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ static float atomicMin(double* address, double val) {
    unsigned long long* address_as_i = (unsigned long long*)address;
    unsigned long long old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed, __double_as_longlong(::fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

class GPUSVM;
__global__ static void train_CUDA_model(GPUSVM model);

// device globals
__device__ number b_up;
__device__ number b_lo;
__device__ idx up;
__device__ idx lo;
class GPUSVM {

    /*  up for:
     * | y_i |   a_i   |
     * |     | 0< a <C |
     * |  1  |    0    |
     * | -1  |    C    |
     * and low for:
     * | y_i |   a_i   |
     * |  1  |    C    |
     * | -1  |    0    |
     */
    enum kind { UP = 1, LO = 0 };

  public:
    cuda_matrix x;
    cuda_vector<label> y;
    vector<number> w;
    Kernel K;
    cuda_vector<number> a;
    cuda_vector<number> error;

    cuda_vector<kind> indexes;
    cuda_vector<number> b;

    number C;
    number tol;      // KKT tolerance
    number diff_tol; // alpha diff tolerance ?

    // cudaLaunchCooperativeKernel
    //  TODO: maybe initialize in host, transfer to cuda_vectors, then train?
    GPUSVM(dataset_shape& shape, matrix& _x, vector<label>& _y, hyperparams params, Kernel kernel)
        : x(_x),
          y(_y),
          w(shape.num_features),
          K(kernel),
          a(shape.num_samples),
          error(shape.num_samples),
          indexes(shape.num_samples),
          b(shape.num_samples),
          C(params.cost),
          tol(params.tolerance),
          diff_tol(params.diff_tolerance) {
        // w.set(1); // WARN: watch out
        w.set(types::epsilon);
    }

    void train() {
        int dev = 0;
        /// This will launch a grid that can maximally fill the GPU, on the default stream with kernel arguments
        int numBlocksPerSm = 0;
        // Number of threads my_kernel will be launched with
        int numThreads = 128;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, train_CUDA_model, numThreads, 0);
        // launch
        GPUSVM* dev_this = nullptr;
        cudaErr(cudaMalloc(&dev_this, sizeof(GPUSVM)));
        cudaErr(cudaMemcpy(dev_this, this, sizeof(GPUSVM), cudaMemcpyHostToDevice));
        void* kernelArgs[] = {this};
        dim3 dimBlock(numThreads, 1, 1);
        dim3 dimGrid(deviceProp.multiProcessorCount * numBlocksPerSm, 1, 1);
        cudaLaunchCooperativeKernel((void*)train_CUDA_model, dimGrid, dimBlock, kernelArgs);
    }

    __device__ void train_device() {
        a.set(0);

        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int stride = blockDim.x * gridDim.x;
        grid_group blocks = this_grid();
        thread_block threads = this_thread_block();

        // initialize indexes
        for (idx i = tid; i < indexes.cols; i += stride) { // :^)
            indexes[i] = ((0 < a[i] && a[i] < C) || (y[i] == 1 && a[i] == 0) || (y[i] == -1 * a[i] == C)) ? UP : LO;
        }
        blocks.sync();

        // initialize error
        // this->error.mutate([this](idx _i) { return this->predict_on(_i) - this->y[_i]; });
        for (idx i = tid; i < error.cols; i += stride) {
            error[i] = -y[i];
        }


        auto [up, lo] = compute_b_up_lo();
        number b_lo = b[lo];
        number b_up = b[up];

        while (b_lo > b_up + 2 * tol) {
            // obtain kIlo,Ilo, kIup,Iup, kIup,Ilo

            // FIX: ? maybe ?
            // all threads run this, but only tid == 0's
            // result is used :|
            number sign = y[up] * y[lo];

            number K_lo_lo = K(x[lo], x[lo]);
            number K_up_up = K(x[up], x[up]);
            number K_up_lo = K(x[up], x[lo]);

            number eta = 2 * K_up_lo - K_up_up - K_lo_lo;

            // update a_I_up , a_I_lo

            number a_up = a[up] - (y[up] * (error[lo] - error[up])) / eta + types::epsilon;

            number a_lo = a[lo] + sign * (a[up] - a_up);

            if (tid == 0) {
                a[lo] = a_lo;
                a[up] = a_up;
            }
            // compute b_up and b_low again
            std::tie(up, lo) = compute_b_up_lo();
            b_lo = b[lo];
            b_up = b[up];
        }
    }

    __device__ std::tuple<idx, idx> compute_b_up_lo() {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int stride = blockDim.x * gridDim.x;
        grid_group blocks = this_grid();
        thread_block threads = this_thread_block();
        // thread locals
        idx local_i_up;
        idx local_i_lo;
        number local_b_up = +types::NUM_MAX;
        number local_b_lo = -types::NUM_MAX;
        // block locals
        __shared__ number block_b_up;
        __shared__ number block_b_lo;
        __shared__ idx block_i_up;
        __shared__ idx block_i_lo;

        // initialize block locals
        if (threadIdx.x == 0) {
            block_b_up = +types::NUM_MAX;
            block_b_lo = -types::NUM_MAX;
        }

        // initialize device globals
        if (tid == 0) {
            b_up = +types::NUM_MAX;
            b_lo = -types::NUM_MAX;
        }
        // find thread local b up and lo
        // TODO: reduction instead of atomics?
        for (idx i = tid; i < b.cols; i += stride) {
            if (indexes[i] == UP) {
                if (error[i] < local_b_up) {
                    local_b_up = b[i];
                    local_i_up = i;
                }
            } else {
                if (error[i] > local_b_lo) {
                    local_b_lo = b[i];
                    local_i_up = i;
                }
            }
        }
        // find block local b up and lo
        atomicMin(&block_b_up, local_b_up);
        atomicMax(&block_b_lo, local_b_lo);

        threads.sync();

        // poor man's atomicArg{Min,Max}
        // basically if the value we atomically found
        // is the one the thread found
        // set the index to the one the thread found
        // very few threads /should/ attempt to write
        // and if they do, any of their results are valid
        if (block_b_up == local_b_up) {
            block_i_up = local_i_up;
        }
        if (block_b_lo == local_b_lo) {
            block_i_lo = local_i_lo;
        }

        threads.sync();

        // block leaders find b up and lo
        if (tid == 0) {
            atomicMin(&b_up, block_b_up);
            atomicMax(&b_lo, block_b_lo);
        }
        blocks.sync();

        // atomicArg v2
        if (tid == 0) {
            if (block_b_up = b_up) {
                up = block_i_up;
            }
            if (block_b_lo = b_lo) {
                lo = block_i_lo;
            }
        }
        blocks.sync();
        return std::make_tuple(UP, LO);
    }

    // __device__ number predict_on(idx i) {
    //     // printf("      predict on %zu\n", i);
    //     return K(w, x[i]) + b;
    // }
    //
    __device__ number predict(vector<number>& sample) {
        // printf("      predict on %zu\n", i);
        return K(w, sample) /*+ b*/; // TODO: actually find b after training is done
    }

    void compute_w() {
        vector<number> host_a = a;
        vector<label> host_y = y;
        matrix host_x = x;
        w.set(0);
        for (idx k = 0; k < w.cols; k++) {
            for (idx i = 0; i < host_a.cols; i++) {
                w[k] += host_a[i] * host_y[i] * host_x[i][k];
            }
        }
    }

    // evaluates the objective function at a point a2
    // W(a1, a2) = a1 + a2
    //           - 1/2 * K11 * a1^2
    //           - 1/2 * K22 * a2^2
    //           - sign* K12 * a1 *a2
    //           - y1 * a1 * v1
    //           - y2 * a2 * v2
    //           + Wconst
    // without loss of generality let the 2 multipliers be a1 and a2
    __device__ number eval_objective_func_at(idx i1, idx i2, number a2) {
        // v_i = \Sum_{j=3}^l y_j a_j^{old} K_ij
        number v_1 = 0;
        for (idx j = 0; j < a.cols; j++) {
            if (j == i1 || j == i2) { // skip i1 and i2
                continue;
            }
            v_1 += y[j] * a[j] * K(x[i1], x[j]);
        }
        number v_2 = 0;
        for (idx j = 0; j < a.cols; j++) {
            if (j == i1 || j == i2) { // skip i1 and i2
                continue;
            }
            v_2 += y[j] * a[j] * K(x[i2], x[j]);
        }

        // constant part of objective function
        // W(a) = \Sum_{i=3}^l a_i
        //      - \frac{1}{2} \Sum_{i=3}{l}\Sum_{j=3}{l} y_i y_j k(x_i, x_j) a_i a_j
        // IDEA: cache it?
        number Wconst = 0;
        for (idx i = 0; i < a.cols; i++) {
            // \Sum_{i=3}^n a_i
            if (i == i1 || i == i2) { // skip i1 and i2
                continue;
            }
            Wconst += a[i];
            // \Sum_{j=3}{l} y_i y_j k(x_i, x_j) a_i a_j
            number inner_sum = 0;
            for (idx j = 0; j < a.cols; j++) {
                if (j == i1 || j == i2) { // skip i1 and i2
                    continue;
                }
                inner_sum += y[i] * y[j] * K(x[i], x[j]) * a[i] * a[j];
            }
            Wconst -= inner_sum / 2;
        }

        // sign
        number s = y[i1] * y[i2];
        // \gamma
        number g = a[i1] + (s * a[i2]);
        // clang-format off
    return g - (s * a2) + a2
        - (K(x[i1], x[i1]) * (g - s * a2))
        / 2
        - (K(x[i2], x[i2]) * a2)
        / 2
        - s * K(i1, i2) * (g - s * a2) * a2
        - y[i1] * (g - s * a2) * v_1
        - y[i2] * a2 * v_2
        + Wconst;
        // clang-format on
    }
};

__global__ static void train_CUDA_model(GPUSVM model) {
    model.train_device();
}

} // namespace SVM
#endif
