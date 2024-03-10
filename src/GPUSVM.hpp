#ifndef GPUSVM_HPP
#define GPUSVM_HPP 1

#include <cmath>
#include <iostream>
#include <iterator>

#include <cooperative_groups.h>

#include "SVM_common.hpp"
#include "dataset.hpp"
#include "matrix.hpp"
#include "types.hpp"
#include "vector.hpp"

#define max(a, b) a > b ? a : b
#define min(a, b) a < b ? a : b

static unsigned int THREADS;
// #define BLOCKS 16
// #define THREADS 128
// #ifndef __CUDA_ARCH__
// namespace cooperative_groups {
// typedef int grid_group;
// typedef int thread_block;
// class this_thread_block {
//     void sync() {
//         assert(("Unreachable", false));
//     }
// };
// class this_grid {
//     void sync() {
//         assert(("Unreachable", false));
//     }
// };
// } // namespace cooperative_groups
// #endif

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

class GPUSVM;
__global__ static void train_CUDA_model(GPUSVM*, size_t);

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
    enum idx_kind { UP = 1, LOW = 0, BOTH = 2 };

  public:
    cuda_matrix<math_t> x;
    cuda_vector<label> y;
    vector<math_t> w;
    cuda_vector<math_t> a;
    cuda_vector<math_t> error;
    cuda_vector<math_t> ddata;
    cuda_vector<idx> dindx;
    // vector<math_t> host_a;
    // vector<label> host_y;
    // matrix<math_t> host_x;

    cuda_vector<idx_kind> indices;
    math_t b;

    math_t C;
    math_t tol;      // KKT tolerance
    math_t diff_tol; // alpha diff tolerance ?
    Kernel_t kernel_type;

    GPUSVM(dataset_shape& shape, matrix<math_t>& _x, vector<label>& _y, hyperparams params, Kernel_t _kernel_type)
        : x(_x),
          y(_y),
          w(shape.num_features),
          a(shape.num_samples),
          error(shape.num_samples),
          ddata(THREADS * 2),
          dindx(THREADS * 2),
          // host_a(0),   // this will be transfered at the end
          // host_y(y), // these get memcpy'ed
          // host_x(x),
          indices(shape.num_samples),
          b(0),
          C(params.cost),
          tol(params.tolerance),
          diff_tol(params.diff_tolerance),
          kernel_type(_kernel_type) {
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

    float train() {

        cudaEvent_t start, stop;
        float time;

        int dev = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        int supportsCoopLaunch = 0;
        cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
		if (!supportsCoopLaunch) {
			fprintf(stderr, "Cooperative kernels are not supported on this hardware\n!");
		}
        /// This will launch a grid that can maximally fill the GPU, on the default stream with kernel arguments
        int numBlocksPerSm = 0;
        // Number of threads my_kernel will be launched with
        cudaGetDeviceProperties(&deviceProp, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, train_CUDA_model, THREADS, 0);
        dim3 dimBlock(THREADS, 1, 1);
        dim3 dimGrid(deviceProp.multiProcessorCount * numBlocksPerSm, 1, 1);
		printf("Using %u blocks!\n", dimGrid.x);

        cudaErr(cudaEventCreate(&start));
        cudaErr(cudaEventCreate(&stop));

        cudaErr(cudaEventRecord(start));
        GPUSVM* dev_this = nullptr;
        // alloc
        cudaErr(cudaMalloc(&dev_this, sizeof(GPUSVM)));
        // move struct to device
        cudaErr(cudaMemcpy(dev_this, this, sizeof(GPUSVM), cudaMemcpyHostToDevice));
        /*
         * trully, whoever thought of this API <my lawyer has advised me not to continue this thought>
         */
        struct Param1 {
            GPUSVM* a;
        };
        struct Param2 {
            size_t b;
        };
        Param1 p1{dev_this};
        Param2 p2{THREADS * 2};
        //  kernelArgs needs to be an array of N pointers, one for each argument of the kernel, we have 2 arguments
        //  so we make 2 structs with our data inside them, and pass their address to kernelArgs
        void* kernelArgs[] = {&p1, &p2};
        cudaErr(cudaLaunchCooperativeKernel((void*)train_CUDA_model, dimGrid, dimBlock, kernelArgs,
                                            THREADS * 2 * sizeof(idx) + THREADS * 2 * sizeof(math_t)));
        cudaErr(cudaMemcpy(this, dev_this, sizeof(GPUSVM), cudaMemcpyDeviceToHost));

        cudaErr(cudaEventRecord(stop));
        cudaErr(cudaEventSynchronize(stop));

        cudaErr(cudaEventElapsedTime(&time, start, stop));
        return time;
    }

    // device globals
    idx dev_lo = 0;
    idx dev_up = 0;
    math_t dev_a_up;
    math_t dev_a_lo;
    math_t dev_a_up_new;
    math_t dev_a_lo_new;
    math_t dev_b_lo = 1;
    math_t dev_b_up = -1;

    __device__ void train_device(size_t shared_memory) {
        a.set(0);

        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int stride = blockDim.x * gridDim.x;
        grid_group blocks = this_grid();
        thread_block threads = this_thread_block();
        // IDEA: block locals ?

        // thread locals
        math_t K_lo_lo;
        math_t K_up_up;
        math_t K_lo_up;
        math_t eta;
        math_t a_lo_new;
        math_t a_up_new;
        math_t a_lo;
        math_t a_up;
        idx lo;
        idx up;

        // initialize indices
        for (idx i = tid; i < indices.cols; i += stride) { // :^)
            if (0 < a[i] && a[i] < C) {
                indices[i] = BOTH;
                continue;
            }
            if ((y[i] == 1 && a[i] == 0) || (y[i] == -1 && a[i] == C)) {
                indices[i] = UP;
            } else {
                indices[i] = LOW;
            }
            // printf("[%d]: indices[%lu] = %s\n", tid, i, indices[i] == UP ? "UP" : indices[i] == LO ? "LO" : "BOTH");
        }

        // initialize error
        // this->error.mutate([this] __device__(idx _i) -> idx { return -this->y[_i]; });
        for (idx i = tid; i < error.cols; i += stride) {
            error[i] = -y[i];
        }
        blocks.sync();

        // pick b_up and _lo for the first time:
        bool picked_lo = false;
        bool picked_up = false;
        if (tid == 0) {
            for (idx i = 0; i < y.cols; i += 1) {
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
                        picked_up = true;
                        if (picked_lo) {
                            break;
                        }
                    }
                }
            }

            K_lo_lo = Kernel(x[lo], x[lo]);
            K_up_up = Kernel(x[up], x[up]);
            K_lo_up = Kernel(x[lo], x[up]);
            eta = K_lo_lo + K_up_up - 2 * K_lo_up;
            math_t a_new = 2 / eta;
            // write to global memory
            dev_a_up_new = a_new;
            dev_a_lo_new = a_new;
            a[lo] = a_new;
            a[up] = a_new;
            dev_lo = lo;
            dev_up = up;

            // recalculate indices for lo and up
            indices[up] = compute_type_for_index(up);
            indices[lo] = compute_type_for_index(lo);
        }
        blocks.sync();
        // read from global
        a_lo_new = dev_a_lo_new;
        a_up_new = dev_a_up_new;
        lo = dev_lo;
        up = dev_up;

        for (idx i = tid; i < error.cols; i += stride) {
            error[i] = error[i] - a_lo_new * Kernel(x[lo], x[i]) + a_up_new * Kernel(x[up], x[i]);
        }

        blocks.sync();
        // int ITERS = 0;
        // int nochange = 0;
        while (dev_b_lo > dev_b_up + 2 * tol) {
            // if (ITERS > 1000) {
            //     printf("Iteration limit reached!\n");
            //     break;
            // }
            // if (nochange > 10) {
            //     printf("No changes to alphas!\n");
            //     break;
            // }
            // ITERS++;

            if (tid == 0) {
                math_t sign = y[up] * y[lo];

                K_lo_lo = Kernel(x[lo], x[lo]);
                K_up_up = Kernel(x[up], x[up]);
                K_lo_up = Kernel(x[lo], x[up]);

                eta = K_lo_lo + K_up_up - 2 * K_lo_up;

                // update a_I_up , a_I_lo

                a_up = a[up];
                a_lo = a[lo];

                a_up_new = a_up + (y[up] * (error[lo] - error[up])) / eta + types::epsilon;

                // clip new a_up
                if (a_up_new > C) {
                    a_up_new = C;
                } else if (a_up_new < 0) {
                    a_up_new = 0;
                }

                a_lo_new = a_lo + sign * (a_up - a_up_new);
                // write to global
                dev_a_up_new = a_up_new;
                dev_a_lo_new = a_lo_new;
            }
            blocks.sync();
            a_lo_new = dev_a_lo_new;
            a_up_new = dev_a_up_new;
            a_lo = dev_a_lo;
            a_up = dev_a_up;

            // if (fabs(a_up_new - a_up) < diff_tol) {
            //     printf("Change = %f num = %d\n", fabs(a_up_new - a_up), nochange);
            //     nochange++;
            // } else {
            //     nochange = 0;
            // }

            // recalculate error
            for (idx i = tid; i < error.cols; i += stride) {
                error[i] = error[i] + (a_lo_new - a_lo) * y[lo] * Kernel(x[lo], x[i]) +
                           (a_up_new - a_up) * y[up] * Kernel(x[up], x[i]);
            }
            blocks.sync();

            if (tid == 0) {
                // set new alphas
                a[lo] = a_lo_new;
                a[up] = a_up_new;
                // recompute index type for up and low
                indices[lo] = compute_type_for_index(lo);
                indices[up] = compute_type_for_index(up);
            }
            blocks.sync();

            // std::tie(up, lo) = compute_b_up_lo();
            auto result = argMin(shared_memory);
            up = result.a;
            lo = result.b;

            if (tid == 0) {
                dev_b_lo = error[lo];
                dev_b_up = error[up];
                // for (idx i = tid; i < a.cols; i += stride) {
                //     printf("a[%lu] = %f\n", i, a[i]);
                // }
                // for (idx i = tid; i < error.cols; i += stride) {
                //     printf("error[%lu] = %f\n", i, error[i]);
                // }
                // for (idx i = tid; i < indices.cols; i += stride) {
                //     printf("[%d]: indices[%lu] = %s\n", tid, i,
                //            indices[i] == UP   ? "UP"
                //            : indices[i] == LO ? "LO"
                //                               : "BOTH");
                // }
                // printf("[%d]: b_lo[%lu] = %f, b_up[%lu] = %f!\t", tid, lo, dev_b_lo, up, dev_b_up);
                // printf("Gap = %f\n", dev_b_lo - (dev_b_up + 2 * tol));
            }
            // printf("[%d]: before\n", tid);
			blocks.sync();
        }
        b = (dev_b_lo + dev_b_up) / 2;
    }

    __device__ void compute_index_types() {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int stride = blockDim.x * gridDim.x;
        grid_group blocks = this_grid();
        thread_block threads = this_thread_block();

        for (idx i = tid; i < indices.cols; i += stride) {
            indices[i] = compute_type_for_index(i);
        }
    }

    __device__ idx_kind compute_type_for_index(idx i) {
        if (0 < a[i] && a[i] < C) {
            return BOTH;
        } else if ((y[i] == 1 && a[i] == 0) || (y[i] == -1 && a[i] == C)) {
            return UP;
        } else {
            return LOW;
        }
    }

    // expects sdata, sindx of size blockDim.x * 2
    //         ddata, dindx of size gridDim.x * 2
    //         must have blockDim.x > gridDim.x
    __device__ idx_tuple argMin(size_t shared_halfpoint) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int stride = blockDim.x * gridDim.x;
        thread_block threads = this_thread_block();
        grid_group blocks = this_grid();

        extern __shared__ char shared_memory[];

        // __shared__ idx sindx[THREADS * 2];
        // __shared__ math_t sdata[THREADS * 2];

        idx* sindx = reinterpret_cast<idx*>(shared_memory);
        math_t* sdata = reinterpret_cast<math_t*>(shared_memory + (sizeof(idx) * shared_halfpoint));

        __shared__ math_t block_max;
        __shared__ idx block_max_i;
        __shared__ math_t block_min;
        __shared__ idx block_min_i;
        // init block locals
        if (threadIdx.x == 0) {
            block_max = -types::MATH_T_MAX;
            block_max_i = 0;
            block_min = +types::MATH_T_MAX;
            block_min_i = 0;
        }

        math_t cur_max;
        math_t cur_min;
        idx min_i = tid;
        idx max_i = tid;
        if (tid < a.cols) {
            cur_min = a[tid];
            // thread local arg min|max
            for (idx i = tid + stride; i < a.cols; i += stride) {
                // TODO: Take into account indices up, low and both
                auto kind = indices[i];
                auto tmp = a[i]; // save to local, so it's accessed only once
                if (kind == UP || kind == BOTH) {
                    if (tmp < cur_min) {
                        cur_min = tmp;
                        min_i = i;
                    }
                }
                if (kind == LOW || kind == BOTH) {
                    if (tmp > cur_min) {
                        cur_max = tmp;
                        max_i = i;
                    }
                }
            }
        } else {
            cur_max = -types::MATH_T_MAX;
            cur_min = +types::MATH_T_MAX;
        }

        // load into to block shared memory
        sdata[threadIdx.x] = cur_min;
        sindx[threadIdx.x] = min_i;
        sdata[threadIdx.x + blockDim.x] = cur_max;
        sindx[threadIdx.x + blockDim.x] = max_i;
        threads.sync();

        // block reduce into sdata[0] and sindx[0]
        for (idx offset = 1; offset < blockDim.x; offset *= 2) {
            idx index = 2 * offset * threadIdx.x;

            if (index < blockDim.x) {
                // min
                if (sdata[index + offset] < sdata[index]) {
                    sdata[index] = sdata[index + offset];
                    sindx[index] = sindx[index + offset];
                }
                // max (stored offset by blockDim.x)
                if (sdata[blockDim.x + index + offset] > sdata[blockDim.x + index]) {
                    sdata[blockDim.x + index] = sdata[blockDim.x + index + offset];
                    sindx[blockDim.x + index] = sindx[blockDim.x + index + offset];
                }
            }
        }

        threads.sync();
        // write block result to device global
        if (threadIdx.x == 0) {
            ddata[blockIdx.x] = sdata[0];
            dindx[blockIdx.x] = sindx[0];
            ddata[blockDim.x + blockIdx.x] = sdata[blockDim.x + 0];
            dindx[blockDim.x + blockIdx.x] = sindx[blockDim.x + 0];
        }

        blocks.sync();

        // perform reduction of block results,
        // like above but for block results :^)
        if (blockIdx.x == 0) {
            // copy device globals to block shared memory
            if (threadIdx.x < gridDim.x) {
                sdata[threadIdx.x] = ddata[threadIdx.x];
                sindx[threadIdx.x] = dindx[threadIdx.x];
                sdata[blockDim.x + threadIdx.x] = ddata[blockDim.x + threadIdx.x];
                sindx[blockDim.x + threadIdx.x] = dindx[blockDim.x + threadIdx.x];
            }
            threads.sync();
            for (idx offset = 1; offset < blockDim.x; offset *= 2) {
                idx index = 2 * offset * threadIdx.x;

                if (index < blockDim.x) {
                    if (sdata[index + offset] < sdata[index]) {
                        sdata[index] = sdata[index + offset];
                        sindx[index] = sindx[index + offset];
                    }
                    if (sdata[blockDim.x + index + offset] > sdata[blockDim.x + index]) {
                        sdata[blockDim.x + index] = sdata[blockDim.x + index + offset];
                        sindx[blockDim.x + index] = sindx[blockDim.x + index + offset];
                    }
                }
            }
            threads.sync();
        }

        // write to global memory
        if (blockIdx.x + threadIdx.x == 0) { // will the real tid 0 plz stand up
            dindx[0] = sindx[0];
            dindx[1] = sindx[blockDim.x + 0];
        }

        blocks.sync();

        return {.a = dindx[0], .b = dindx[1]};
    }

    // for 1 thread, sanity check of argmin/max
    // __device__ std::tuple<idx, idx> compute_b_up_lo() {
    //     math_t b_up = +types::MATH_T_MAX;
    //     math_t b_lo = -types::MATH_T_MAX;
    //     idx i_up = 0;
    //     idx i_lo = 0;
    //
    //     // find thread local b up and lo
    //     // TODO: reduction instead of atomics?
    //     for (idx i = 0; i < error.cols; i += 1) {
    //         if (indices[i] == BOTH) {
    //             if (error[i] < b_up) {
    //                 b_up = error[i];
    //                 i_up = i;
    //                 // printf("new up %f\n", b_up);
    //             }
    //             if (error[i] > b_lo) {
    //                 b_lo = error[i];
    //                 i_lo = i;
    //                 // printf("new lo %f\n", b_lo);
    //             }
    //         } else if (indices[i] == UP) {
    //             // printf("[%d]: err[%lu]=%f b_up=%f\n", tid, i, error[i], b_up);
    //             if (error[i] < b_up) {
    //                 b_up = error[i];
    //                 i_up = i;
    //                 // printf("new up %f\n", b_up);
    //             }
    //         } else {
    //             // printf("[%d]: err[%lu]=%f b_lo=%f\n", tid, i, error[i], b_lo);
    //             if (error[i] > b_lo) {
    //                 b_lo = error[i];
    //                 i_lo = i;
    //                 // printf("new lo %f\n", b_lo);
    //             }
    //         }
    //     }
    //     return std::make_tuple(i_up, i_lo);
    // }

    // __device__ math_t predict_on(idx i) {
    //     // printf("      predict on %zu\n", i);
    //     return K(w, x[i]) + b;
    // }
    //
    __device__ __host__ math_t predict(vector<math_t>& sample) {
        // printf("      predict on %zu\n", i);
        return Kernel(w, sample); // TODO: actually find b after training is done
                                  // math_t res = 0;
                                  // for (idx i = 0; i < x.rows; i++) {
                                  // 	res += host_a[i] * host_y[i] * Kernel(host_x[i], sample) + b;
                                  // }
                                  // return res;
    }

    void compute_w() {
        // if (host_a.cols == 0) {
        // 	host_a = a; // cudaMemcpy'ed
        // }
        vector<math_t> host_a = a;
        vector<label> host_y = y;
        matrix<math_t> host_x = x;
        w.set(0);
        for (idx k = 0; k < w.cols; k++) {
            for (idx i = 0; i < host_a.cols; i++) {
                w[k] += host_a[i] * host_y[i] * host_x[i][k];
            }
        }
    }
};

__global__ static void train_CUDA_model(GPUSVM* model, size_t shared_memory) {
    model->train_device(shared_memory);
}

} // namespace SVM
#endif

// __device__ static double atomicMax(double* address, double val) {
//     unsigned long long* address_as_i = (unsigned long long*)address;
//     unsigned long long old = *address_as_i, assumed;
//     do {
//         assumed = old;
//         old = ::atomicCAS(address_as_i, assumed, __double_as_longlong(::fmax(val, __longlong_as_double(assumed))));
//     } while (assumed != old);
//     return __longlong_as_double(old);
// }
//
// __device__ static double atomicMin(double* address, double val) {
//     unsigned long long* address_as_i = (unsigned long long*)address;
//     unsigned long long old = *address_as_i, assumed;
//     do {
//         assumed = old;
//         old = ::atomicCAS(address_as_i, assumed, __double_as_longlong(::fmin(val, __longlong_as_double(assumed))));
//     } while (assumed != old);
//     return __longlong_as_double(old);
// }
