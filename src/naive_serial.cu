

#include <cmath>
#include <cstdio>

#include "types.hpp"

#include "OVA.hpp"
#include "SMO.hpp"
#include "GPUSVM.hpp"
#include "cuda_helpers.h"
#include "dataset.hpp"
#include "vector.hpp"

using std::printf;
using SVM::GPUSVM;
using SVM::SMO;
using types::base_vector;
using types::cuda_vector;
using types::idx;
using types::Kernel;
using types::label;
using types::math_t;
using types::vector;

int main(void) {
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
    // dataset data(8, 100, xor_data, ',');
    auto lin_data = std::fopen("../datasets/linear.data", "r");
    if (!lin_data) {
        printf("Missing dataset!\n");
        exit(1);
    }
    dataset data(3, 1000, lin_data, ';');

    // number (*t)(vector<number>, vector<number>) = Polynomial_Kernel<>;

    // SVM::OVA<SMO> ova(data.shape, data.X, data.Y, {.cost = 3, .tolerance = 3e-3, .diff_tolerance = 1e-4}, SVM::RBF);

    // SVM::OVA<SMO> ova(data.shape, data.X, data.Y, {.cost = 3, .tolerance = 3e-3, .diff_tolerance = 0.01}, SVM::LINEAR);
    SVM::OVA<GPUSVM> ova(data.shape, data.X, data.Y, {.cost = 3, .tolerance = 3e-3, .diff_tolerance = 0.01}, SVM::LINEAR);
    // SVM::OVA<SMO> ova(data.shape, data.X, data.Y, {.cost = 30, .tolerance = 1e-6, .diff_tolerance = 0.001}, SVM::POLY);
    // SVM::OVA<GPUSVM> ova(data.shape, data.X, data.Y, {.cost = 3, .tolerance = 1e-3, .diff_tolerance = 1e-6}, SVM::POLY);
    // SVM::OVA ova(data.shape, data.X, data.Y, { .cost = 300, .tolerance = 3e-1, .diff_tolerance =  0.000000001 },
    // RBF_Kernel);
    ova.train();
    ova.test(data);
    // SVM::SVM model(data, {.cost = COST, .tolerance = TOL, .diff_tolerance = 0.01}, Linear_Kernel);
    // printd(model.w);
    // model.test();

    return 0;
}
