

#include <cmath>
#include <cstdio>

#include "argparse/include/argparse/argparse.hpp"
#include "types.hpp"

#include "GPUSVM.hpp"
#include "OVA.hpp"
#include "SMO.hpp"
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

constexpr unsigned int hash(const char* s, int off = 0) {
    return !s[off] ? 5381 : (hash(s, off + 1) * 33) ^ s[off];
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("svm");

    // program.add_argument("square").help("display the square of a given integer").scan<'i', int>();
    program.add_argument("dataset").default_value(std::string{"linear"}).help("the dataset to use").metavar("DATASET");
    program.add_argument("method").default_value(std::string{"cpu"}).help("algorithm to use").metavar("ALGO");
    program.add_argument("--threads").default_value(unsigned{128}).help("number of threads for CUDA").scan<'i', unsigned>();
    program.add_argument("--size").default_value(size_t{1000}).help("size of the linear DATASET").scan<'i', size_t>();
    program.add_argument("--test").help("test the model after training").flag();

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << "Error during argument parsing:" << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    THREADS = program.get<unsigned>("--threads");
    bool test = program.get<bool>("--test");

    // auto input = program.get<int>("square");

    std::string filepath = "";
    size_t feats = 0;
    size_t rows = 0;
    char delim = '\0';
    bool header = false;

    switch (hash(program.get("dataset").c_str())) {
    case hash("xor"):
        filepath = "../datasets/xor.data";
        feats = 8;
        rows = 10000;
        delim = ',';
        break;
    case hash("iris"):
        filepath = "../datasets/iris.data";
        feats = 4;
        rows = 150;
        delim = ',';
        break;
    case hash("wine"):
        filepath = "../datasets/winequality-red.csv";
        feats = 11;
        rows = 1599;
        delim = ';';
        header = true;
        break;
    case hash("linear"):
        feats = 3;
        rows = program.get<size_t>("size");
        delim = ';';
        switch (rows) {
        case 1000:
            filepath = "../datasets/linear1k.data";
            break;
        case 10000:
            filepath = "../datasets/linear10k.data";
            break;
        case 100000:
            filepath = "../datasets/linear100k.data";
            break;
        case 1000000:
            filepath = "../datasets/linear1M.data";
            break;
        case 10000000:
            filepath = "../datasets/linear10M.data";
            break;
        default:
            printf("size must be in (1000, 10.000, 100.000, 1.000.000)");
            exit(-1);
        }
        break;
    case hash("diabetes"):
        printf("Not ready yet: dataset: %s\n", argv[1]);
        exit(-1);
        break;
    default:
        printf("No such dataset: %s\n", argv[1]);
        exit(-1);
    }

    auto file = std::fopen(filepath.c_str(), "r");
	if (!file) {
		printf("Error opening file: %s", filepath.c_str());
		exit(-1);
	}
    dataset data(feats, rows, file, delim, header);

    switch (hash(program.get("method").c_str())) {
    case hash("cpu"): {
        SVM::OVA<SMO> ova(data.shape, data.X, data.Y, {.cost = 3, .tolerance = 3e-3, .diff_tolerance = 0.01},
                          SVM::POLY);
        float time = ova.train();
        if (test) {
            ova.test(data);
        }
        printf("Took %f seconds!\n", static_cast<double>(time));
        break;
    }
    case hash("gpu"): {
        SVM::OVA<GPUSVM> ova(data.shape, data.X, data.Y, {.cost = 3, .tolerance = 3e-3, .diff_tolerance = 0.01},
                             SVM::LINEAR);
        ova.train();
        float time = ova.train();
        if (test) {
            ova.test(data);
        }
        printf("Took %f seconds!\n", static_cast<double>(time));
        break;
    }
    default: {
        printf("No such ALGO: '%s'", program.get("method").c_str());
    }
    }

    return 0;
}
// number (*t)(vector<number>, vector<number>) = Polynomial_Kernel<>;

// SVM::OVA<SMO> ova(data.shape, data.X, data.Y, {.cost = 3, .tolerance = 3e-3, .diff_tolerance = 1e-4}, SVM::RBF);

// SVM::OVA<SMO> ova(data.shape, data.X, data.Y, {.cost = 3, .tolerance = 3e-3, .diff_tolerance = 0.01},
// SVM::LINEAR);
// SVM::OVA<GPUSVM> ova(data.shape, data.X, data.Y, {.cost = 3, .tolerance = 3e-3, .diff_tolerance = 0.01},
//                      SVM::LINEAR);
// SVM::OVA<SMO> ova(data.shape, data.X, data.Y, {.cost = 30, .tolerance = 1e-6, .diff_tolerance = 0.001},
// SVM::POLY); SVM::OVA<GPUSVM> ova(data.shape, data.X, data.Y, {.cost = 3, .tolerance = 1e-3, .diff_tolerance =
// 1e-6}, SVM::POLY); SVM::OVA ova(data.shape, data.X, data.Y, { .cost = 300, .tolerance = 3e-1, .diff_tolerance =
// 0.000000001 }, RBF_Kernel);
// ova.train();
// ova.test(data);
// SVM::SVM model(data, {.cost = COST, .tolerance = TOL, .diff_tolerance = 0.01}, Linear_Kernel);
// printd(model.w);
// model.test();

// auto wine = std::fopen("../datasets/winequality-red.csv", "r");
// if (!wine) {
//     printf("Missing dataset!\n");
//     exit(1);
// }
// // dataset data(11, 1599, wine, ';', true);
//
// auto iris = std::fopen("../datasets/iris.data", "r");
// if (!iris) {
//     printf("Missing dataset!\n");
//     exit(1);
// }
// // dataset data(4, 150, iris, ',');
//
// auto xor_data = std::fopen("../datasets/xor.data", "r");
// if (!xor_data) {
//     printf("Missing dataset!\n");
//     exit(1);
// }
// // dataset data(8, 100, xor_data, ',');
// auto lin_data = std::fopen("../datasets/linear.data", "r");
// if (!lin_data) {
//     printf("Missing dataset!\n");
//     exit(1);
// }
// dataset data(3, 1000, lin_data, ';');
