
#include "dataset.hpp"
#include "types.hpp"
#include <cstdio>

int main() {

    // auto file = std::fopen("../datasets/iris.data", "r");
    // dataset<float> data(4, 150, file);
    // data.printX();
    types::vector<double, false, false> a(10);
    types::matrix<double> b(10, 2);
    b[1][0] = 5;

}
