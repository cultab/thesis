

#include <cmath>
#include <cstdio>

#include "types.hpp"

#include "OVA.hpp"
#include "SVM_serial.hpp"
#include "cuda_helpers.h"
#include "dataset.hpp"
#include "vector.hpp"

using std::printf;
using types::cuda_vector;
using types::idx;
using types::Kernel;
using types::label;
using types::number;
using types::vector;

number Linear_Kernel(vector<number> a, vector<number> b) {
    number res = 0;
    for (idx k = 0; k < a.cols; k++) {
        res += a[k] * b[k];
    }
    return res;
}

// for xor -> degree = 3, gamma = 0
number Polynomial_Kernel(vector<number> a, vector<number> b) {
    number gamma = 0;
    number degree = 3;
    number res = 0;
    for (idx k = 0; k < a.cols; k++) {
        res += a[k] * b[k];
    }

    return gamma + pow(res, degree);
}

// TODO: this don't work :|
number RBF_Kernel(vector<number> a, vector<number> b) {
    assert(false && "this don't work fren");
    number gamma = 0.01;
    number res = 0;
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

__global__ void testKernel(cuda_vector<int> a) {
    unsigned int tid = threadIdx.x;
    printf("henlo from %d\n", tid);
    // printf("nice = %d\n", a);
    // a = 5;
    // printf("nice = %d\n", a);
    printf("henlo %p\n", a.cols);
    printf("%d\n", a.cols);
    if (tid < a.cols) {
        printf("%d\n", a.data[tid]);
    } else {
        printf("%d\n", a.cols);
    }
}

int main(void) {
    // vector<int> h(3);
    // h[0] = 1;
    // h[1] = 2;
    // h[2] = 3;
    // cuda_vector<int> d2(h);
    // testKernel<<<1, 3>>>(d2);
    // puts("here");
    // cudaLastErr();
    // exit(0);
    vector<number> a(3);
    a[0] = 0.3;
    a[1] = 1.7;
    a[2] = -3.7;
    vector<number> b(3);
    b[0] = -1.3;
    b[1] = 2.5;
    b[2] = -3.7;
    vector<number> c(3);
    c[0] = -1.3;
    c[1] = 2.4;
    c[2] = -3.7;
    // auto res = RBF_Kernel(a, a);
    auto res = RBF_Kernel(b, a);
    printf("res = %.25f\n", res);
    // exit(0);

    auto wine = std::fopen("../datasets/winequality-red.csv", "r");
    if (!wine) {
        printf("Missing dataset!\n");
        exit(1);
    }
    // dataset data(11, 1599, wine, ';', true);

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
    dataset data(8, 100, xor_data, ',');

    // number (*t)(vector<number>, vector<number>) = Polynomial_Kernel<>;

    // SVM::OVA ova(data.shape, data.X, data.Y, {.cost = 3, .tolerance = 3e-3, .diff_tolerance = 0.01}, Linear_Kernel);
    SVM::OVA ova(data.shape, data.X, data.Y, {.cost = 30, .tolerance = 3e-3, .diff_tolerance = 0.001}, Polynomial_Kernel);
    // SVM::OVA ova(data.shape, data.X, data.Y, { .cost = 300, .tolerance = 3e-1, .diff_tolerance =  0.000000001 }, RBF_Kernel);
    ova.train();
    getchar();
    ova.test(data);
    // SVM::SVM model(data, {.cost = COST, .tolerance = TOL, .diff_tolerance = 0.01}, Linear_Kernel);
    // printd(model.w);
    // model.test();

    return 0;
}
