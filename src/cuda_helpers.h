#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H 1
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

inline void cudaPrintError(cudaError_t cudaerr, const char* file, int line) {
    if (cudaerr != cudaSuccess) {
        fprintf(stderr, "CUDA error: \"%s\" in file %s at line %d.\n", cudaGetErrorString(cudaerr), file, line);
        exit(cudaerr);
    }
}

inline void printError() {}

#define err(ans, msg)                                                                                                  \
    do {                                                                                                               \
        if (ans == nullptr) {                                                                                          \
            fprintf(stderr, "error: \"%s\" in file %s at line %d.\n", msg, __FILE__, __LINE__)                         \
        }                                                                                                              \
    } while (0)

#define cudaErr(ans)                                                                                                   \
    do {                                                                                                               \
        cudaPrintError((ans), __FILE__, __LINE__);                                                                     \
    } while (0)

#define cudaLastErr()                                                                                                  \
    do {                                                                                                               \
        cudaError_t cudaerr = cudaDeviceSynchronize();                                                                 \
        cudaPrintError(cudaerr, __FILE__, __LINE__);                                                                   \
    } while (0)

#endif
