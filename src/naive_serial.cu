

#include <cmath>
#include <cstdio>

#include "types.hpp"

#include "GPUSVM.hpp"
#include "OVA.hpp"
// #include "SMO.hpp"
#include "cuda_helpers.h"
#include "dataset.hpp"
#include "vector.hpp"

using std::printf;
// using SVM::SMO;
using SVM::GPUSVM;
using types::base_vector;
using types::cuda_vector;
using types::idx;
using types::Kernel;
using types::label;
using types::math_t;
using types::vector;

math_t Linear_Kernel(base_vector<math_t> a, base_vector<math_t> b) {
    math_t res = 0;
    for (idx k = 0; k < a.cols; k++) {
        res += a[k] * b[k];
    }
    return res;
}

// for xor -> degree = 3, gamma = 0
math_t Polynomial_Kernel(base_vector<math_t> a, base_vector<math_t> b) {
    math_t gamma = 0;
    math_t degree = 3;
    math_t res = 0;
    for (idx k = 0; k < a.cols; k++) {
        res += a[k] * b[k];
    }

    return gamma + pow(res, degree);
}

// TODO: this don't work :|
math_t RBF_Kernel(base_vector<math_t> a, base_vector<math_t> b) {
    assert(false && "this don't work fren");
    math_t gamma = 0.01;
    math_t res = 0;
    for (idx k = 0; k < a.cols; k++) {
        res += pow(a[k] - b[k], 2); // squared euclidean distance -> aka sum of square of difference
    }

    // printf("inres = %f\n", res);
    res *= -gamma;
    // printf("inres = %f\n", res);
    res = exp(res);
    // printf("inres = %f\n", res);
    return res;
}

__global__ void testKernel(cuda_vector<int> &a) {
    unsigned int tid = threadIdx.x;
    printf("henlo from %d\n", tid);
    // printf("nice = %d\n", a);
    // a = 5;
    // printf("nice = %d\n", a);
    // printf("%d\n", a.cols);
    if (tid < a.cols) {
        printf("%d\n", a.data[tid]);
        a[tid] = 21;
    }
}

__global__ void testKernel2(int &a) {
    unsigned int tid = threadIdx.x;
    printf("henlo from %d\n", tid);
    // printf("nice = %d\n", a);
    // a = 5;
    // printf("nice = %d\n", a);
    // printf("%d\n", a.cols);
    if (tid == 0) {
        a = 9;
    }
}

int main(void) {
    // vector<int> h(4);
    // h[0] = 1;
    // h[1] = 2;
    // h[2] = 3;
    // cuda_vector<int> d2(h);
    // testKernel<<<1, 5>>>(d2);
    // testKernel<<<1, 5>>>(d2);
    // cudaLastErr();
    // vector<int> h2 = d2;
    // printf("back out %d\n", h2[1]);
    // exit(0);
    // puts("here");
    // cudaLastErr();
    // exit(0);
    // vector<number> a(3);
    // a[0] = 0.3;
    // a[1] = 1.7;
    // a[2] = -3.7;
    // vector<number> b(3);
    // b[0] = -1.3;
    // b[1] = 2.5;
    // b[2] = -3.7;
    // vector<number> c(3);
    // c[0] = -1.3;
    // c[1] = 2.4;
    // c[2] = -3.7;
    // auto res = RBF_Kernel(a, a);
    // auto res = RBF_Kernel(b, a);
    // printf("res = %.25f\n", res);
    // exit(0);

    auto wine = std::fopen("../datasets/winequality-red.csv", "r");
    if (!wine) {
        printf("Missing dataset!\n");
        exit(1);
    }
    dataset data(11, 1599, wine, ';', true);

    auto iris = std::fopen("../datasets/iris.data", "r");
    if (!iris) {
        printf("Missing dataset!\n");
        exit(1);
    }
    // dataset data(4, 150, iris, ',');

    auto xor_data = std::fopen("../datasets/xor.data", "r");
    if (!xor_data) {
        printf("Missing dataset!\n");
        exit(1);
    }
    // dataset data(8, 100, xor_data, ',');

    // number (*t)(vector<number>, vector<number>) = Polynomial_Kernel<>;

    // SVM::OVA<GPUSVM> ova(data.shape, data.X, data.Y, {.cost = 3, .tolerance = 3e-3, .diff_tolerance = 0.01}, SVM::LINEAR);
    SVM::OVA<GPUSVM> ova(data.shape, data.X, data.Y, {.cost = 30, .tolerance = 1e-6, .diff_tolerance = 0.001}, SVM::POLY);
    // SVM::OVA ova(data.shape, data.X, data.Y, { .cost = 300, .tolerance = 3e-1, .diff_tolerance =  0.000000001 },
    // RBF_Kernel);
    ova.train();
    puts("Waiting for input..");
    ova.test(data);
    // SVM::SVM model(data, {.cost = COST, .tolerance = TOL, .diff_tolerance = 0.01}, Linear_Kernel);
    // printd(model.w);
    // model.test();

    return 0;
}
