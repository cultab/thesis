
#include "dataset.hpp"
#include "types.hpp"
#include <fstream>
#include <sstream>

int main() {

    std::ifstream file("../datasets/iris.data");
    dataset<int> data(5, 150, file);
    // data.printX();
}
