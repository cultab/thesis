
#include "dataset.hpp"
#include "types.hpp"

int main() {

    // auto file = std::fopen("../datasets/iris.data", "r");
    // dataset<float> data(4, 150, file);
    // data.printX();
    types::vector<double, false> a(10);
    a[2] = 5;
    types::matrix<double> b(10, 2);
    b[1][0] = 5;

    return b[1][0] + a[2];
}
