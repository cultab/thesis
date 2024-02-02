#include <iterator>
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
using types::base_vector;
using types::cuda_matrix;
using types::cuda_vector;
using types::idx;
using types::Kernel;
using types::label;
using types::math_t;
using types::matrix;
using types::vector;

namespace SVM {

__host__ __device__ math_t Linear_Kernel(base_vector<math_t> a, base_vector<math_t> b) {
    math_t res = 0;
    for (idx k = 0; k < a.cols; k++) {
        res += a[k] * b[k];
    }
    return res;
}

// for xor -> degree = 3, gamma = 0
__host__ __device__ math_t Polynomial_Kernel(base_vector<math_t> a, base_vector<math_t> b) {
    math_t gamma = 0;
    math_t degree = 3;
    math_t res = 0;
    for (idx k = 0; k < a.cols; k++) {
        res += a[k] * b[k];
    }

    return gamma + pow(res, degree);
}

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
__global__ static void train_CUDA_model(GPUSVM* model);

class GPUSVM {

    /*
     * I0 : 0<a<C any y
     * I1 : y=+1 a=0
     * I2 : y=-1 a=C
     * I3 : y=+1 a=C
     * I4 : y=-1 a=0
     *
     * Ilo : I0 ∪ I3 ∪ I4
     * Iup : I0 ∪ I1 ∪ I2
     */

    /*  up for:
     * | y_i |   a_i   |
     * |  1  |    0    |
     * | -1  |    C    |
     * and low for:
     * | y_i |   a_i   |
     * |  1  |    C    |
     * | -1  |    0    |
     * and both for:
     * | y_i |   a_i   |
     * |     | 0< a <C |
     */
    enum kind { UP = 1, LO = 0, BOTH = 2 };

  public:
    cuda_matrix<math_t> x;
    cuda_vector<label> y;
    vector<math_t> w;
    Kernel_t kernel_type;
    cuda_vector<math_t> a;
    cuda_vector<math_t> error;

    cuda_vector<kind> indices;
    math_t b;

    math_t C;
    math_t tol;      // KKT tolerance
    math_t diff_tol; // alpha diff tolerance ?

    // cudaLaunchCooperativeKernel
    //  TODO: maybe initialize in host, transfer to cuda_vectors, then train?
    GPUSVM(dataset_shape& shape, matrix<math_t>& _x, vector<label>& _y, hyperparams params, Kernel_t _kernel_type)
        : x(_x),
          y(_y),
          w(shape.num_features),
          kernel_type(_kernel_type),
          a(shape.num_samples),
          error(shape.num_samples),
          indices(shape.num_samples),
          b(0),
          C(params.cost),
          tol(params.tolerance),
          diff_tol(params.diff_tolerance) {
        // w.set(1); // WARN: watch out
        w.set(types::epsilon);
    }

    // TODO: caching
    __host__ __device__ math_t Kernel(base_vector<math_t> v, base_vector<math_t> u) {
        switch (kernel_type) {
        case LINEAR:
            return Linear_Kernel(v, u);
            break;
        case POLY:
            return Polynomial_Kernel(v, u);
            break;
        case RBF:
        default:
            printf("UNIMPLEMENTED KERNEL TYPE!\n");
            return 0;
        }
    }

    void train() {
        int dev = 0;
        /// This will launch a grid that can maximally fill the GPU, on the default stream with kernel arguments
        int numBlocksPerSm = 0;
        // number of threads my_kernel will be launched with
        int numThreads = 128;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, train_CUDA_model, numThreads, 0);
        // launch
        GPUSVM* dev_this = nullptr;
        cudaErr(cudaMalloc(&dev_this, sizeof(GPUSVM)));
        cudaErr(cudaMemcpy(dev_this, this, sizeof(GPUSVM), cudaMemcpyHostToDevice));
        dim3 dimBlock(numThreads, 1, 1);
        dim3 dimGrid(deviceProp.multiProcessorCount * numBlocksPerSm, 1, 1);
        /*********************
         * Nvidia, fuck you. *
         *********************/
        struct Params {
            GPUSVM* a;
        };
        Params p{dev_this};
        void* kernelArgs[] = {&p};
        // cudaLaunchCooperativeKernel((void*)train_CUDA_model, dimGrid, dimBlock, kernelArgs);
        cudaLaunchCooperativeKernel((void*)train_CUDA_model, 1, 1, kernelArgs);
        cudaLastErr();
        cudaErr(cudaMemcpy(this, dev_this, sizeof(GPUSVM), cudaMemcpyDeviceToHost));
    }

    __device__ void train_device() {
        a.set(0);

        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int stride = blockDim.x * gridDim.x;
        grid_group blocks = this_grid();
        thread_block threads = this_thread_block();

        if (tid == 0) {
            printf("[%d]: here!\n", tid);
        }

        // initialize indices
        for (idx i = tid; i < indices.cols; i += stride) { // :^)
            if (0 < a[i] && a[i] < C) {
                indices[i] = BOTH;
                continue;
            }
            if ((y[i] == 1 && a[i] == 0) || (y[i] == -1 && a[i] == C)) {
                indices[i] = UP;
            } else {
                indices[i] = LO;
            }
            // printf("[%d]: indices[%lu] = %s\n", tid, i, indices[i] == UP ? "UP" : indices[i] == LO ? "LO" : "BOTH");
        }

        // initialize error
        // this->error.mutate([this] __device__(idx _i) -> idx { return -this->y[_i]; });
        for (idx i = tid; i < error.cols; i += stride) {
            error[i] = -y[i];
        }
        blocks.sync();

        // pick b up and lo for the first time:
        bool picked_lo = false;
        bool picked_up = false;
        idx lo = 0;
        idx up = 0;
        math_t b_lo = 1;
        math_t b_up = -1;
        for (int i = 0; i < y.cols; i++) {
            if (y[i] < 0) {
                if (!picked_lo) {
                    lo = i;
                    picked_lo = true;
                    if (picked_up) {
                        break;
                    }
                }
            } else {
                if (!picked_up) {
                    up = i;
                    if (picked_lo) {
                        break;
                    }
                }
            }
        }

        math_t K_lo_lo = Kernel(x[lo], x[lo]);
        math_t K_up_up = Kernel(x[up], x[up]);
        math_t K_lo_up = Kernel(x[lo], x[up]);
        math_t eta = K_lo_lo + K_up_up - 2 * K_lo_up;
        math_t a_up_new = 2 / eta;
        math_t a_lo_new = a_up_new;

        a[lo] = a_lo_new;
        a[up] = a_up_new;

        // recalculate indices for lo and up
        if (0 < a[up] && a[up] < C) {
            indices[up] = BOTH;
        } else if ((y[up] == 1 && a[up] == 0) || (y[up] == -1 && a[up] == C)) {
            indices[up] = UP;
        } else {
            indices[up] = LO;
        }

        if (0 < a[lo] && a[lo] < C) {
            indices[lo] = BOTH;
        } else if ((y[lo] == 1 && a[lo] == 0) || (y[lo] == -1 && a[lo] == C)) {
            indices[lo] = UP;
        } else {
            indices[lo] = LO;
        }

        for (idx i = tid; i < error.cols; i += stride) {
            error[i] = error[i] - a_lo_new * Kernel(x[lo], x[i]) + a_up_new * Kernel(x[up], x[i]);
        }

        if (tid == 0) {
            printf("[%d]: %s!\n", tid, b_lo > b_up + 2 * tol ? "true" : "false");
            printf("[%d]: b_lo = %f, b_up = %f!\n", tid, b_lo, b_up);
            printf("[%d]: lo = %lu, up = %lu!\n", tid, lo, up);
            // for (idx i = tid; i < a.cols; i += stride) {
            //     printf("a[%lu] = %f\n", i, a[i]);
            // }
            // for (idx i = tid; i < error.cols; i += stride) {
            //     printf("error[%lu] = %f\n", i, error[i]);
            // }
            // for (idx i = tid; i < indices.cols; i += stride) {
            //     printf("[%d]: indices[%lu] = %s\n", tid, i, indices[i] == UP ? "UP" : indices[i] == LO ? "LO" : "BOTH");
            // }
        }

        while (b_lo > b_up + 2 * tol) {

            // obtain kIlo,Ilo, kIup,Iup, kIup,Ilo

            math_t sign = y[up] * y[lo];

            K_lo_lo = Kernel(x[lo], x[lo]);
            K_up_up = Kernel(x[up], x[up]);
            K_lo_up = Kernel(x[lo], x[up]);

            eta = K_lo_lo + K_up_up - 2 * K_lo_up;

            // update a_I_up , a_I_lo

            math_t a_up = a[up];
            math_t a_lo = a[lo];

            a_up_new = a_up + (y[up] * (error[lo] - error[up])) / eta + types::epsilon;

            a_lo_new = a_lo + sign * (a_up - a_up_new);

            // recalculate error
            for (idx i = tid; i < error.cols; i += stride) {
                error[i] = error[i] + (a_lo_new - a_lo) * y[lo] * Kernel(x[lo], x[i]) +
                           (a_up_new - a_up) * y[up] * Kernel(x[up], x[i]);
            }

            // set new alphas
            if (tid == 0) {
                a[lo] = a_lo_new;
                a[up] = a_up_new;
            }

            if (0 < a[up] && a[up] < C) {
                indices[up] = BOTH;
            } else if ((y[up] == 1 && a[up] == 0) || (y[up] == -1 && a[up] == C)) {
                indices[up] = UP;
            } else {
                indices[up] = LO;
            }
            if (0 < a[lo] && a[lo] < C) {
                indices[lo] = BOTH;
            } else if ((y[lo] == 1 && a[lo] == 0) || (y[lo] == -1 && a[lo] == C)) {
                indices[lo] = UP;
            } else {
                indices[lo] = LO;
            }

            printf("BEFORE\n");
            std::tie(up, lo) = compute_b_up_lo();
            printf("AFTER\n");
            b_lo = error[lo];
            b_up = error[up];
            if (tid == 0) {
                // for (idx i = tid; i < a.cols; i += stride) {
                //     printf("a[%lu] = %f\n", i, a[i]);
                // }
                for (idx i = tid; i < error.cols; i += stride) {
                    printf("error[%lu] = %f\n", i, error[i]);
                }
                // for (idx i = tid; i < indices.cols; i += stride) {
                //     printf("[%d]: indices[%lu] = %s\n", tid, i,
                //            indices[i] == UP   ? "UP"
                //            : indices[i] == LO ? "LO"
                //                               : "BOTH");
                // }
                printf("[%d]: %s!\n", tid, b_lo > b_up + 2 * tol ? "true" : "false");
                printf("[%d]: b_lo = %f, b_up = %f!\n", tid, b_lo, b_up);
                printf("[%d]: lo = %lu, up = %lu!\n", tid, lo, up);
            }
        }
    }

    // for 1 thread, sanity check of argmin/max
    __device__ std::tuple<idx, idx> compute_b_up_lo() {
        math_t b_up = +types::MATH_T_MAX;
        math_t b_lo = -types::MATH_T_MAX;
        idx i_up = 0;
        idx i_lo = 0;

        // find thread local b up and lo
        // TODO: reduction instead of atomics?
        for (idx i = 0; i < error.cols; i += 1) {
            if (indices[i] == BOTH) {
                if (error[i] < b_up) {
                    b_up = error[i];
                    i_up = i;
                    printf("new up %f\n", b_up);
                }
                if (error[i] > b_lo) {
                    b_lo = error[i];
                    i_lo = i;
                    printf("new lo %f\n", b_lo);
                }
            } else if (indices[i] == UP) {
                // printf("[%d]: err[%lu]=%f b_up=%f\n", tid, i, error[i], b_up);
                if (error[i] < b_up) {
                    b_up = error[i];
                    i_up = i;
                    printf("new up %f\n", b_up);
                }
            } else {
                // printf("[%d]: err[%lu]=%f b_lo=%f\n", tid, i, error[i], b_lo);
                if (error[i] > b_lo) {
                    b_lo = error[i];
                    i_lo = i;
                    printf("new lo %f\n", b_lo);
                }
            }
        }
        return std::make_tuple(i_up, i_lo);
    }

    // __device__ math_t predict_on(idx i) {
    //     // printf("      predict on %zu\n", i);
    //     return K(w, x[i]) + b;
    // }
    //
    __device__ __host__ math_t predict(vector<math_t>& sample) {
        // printf("      predict on %zu\n", i);
        return Kernel(w, sample) /*+ b*/; // TODO: actually find b after training is done
    }

    void compute_w() {
        vector<math_t> host_a = a;
        vector<label> host_y = y;
        matrix<math_t> host_x = x;
        // printd(host_a);
        // printd(host_y);
        // printd(host_x);
        w.set(0);
        for (idx k = 0; k < w.cols; k++) {
            for (idx i = 0; i < host_a.cols; i++) {
                // printf("a %f\n", host_a[i]);
                // printf("y %d\n", host_y[i]);
                // printf("x %f\n", host_x[i][k]);
                // printf("i %lu, k %lu\n", i, k);
                // printf("product %f\n", host_a[i] * host_y[i] * host_x[i][k]);
                w[k] += host_a[i] * host_y[i] * host_x[i][k];
            }
        }
    }
};

__global__ static void train_CUDA_model(GPUSVM* model) {
    model->train_device();
}

} // namespace SVM
#endif
