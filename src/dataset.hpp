#ifndef DATASET_H
#define DATASET_H

#include "types.hpp"

#include <iostream>
#include <istream>
#include <map>

using std::string;
using std::stringstream;
using std::cout, std::endl;

template <typename number> class dataset {
    using vector = types::vector<number>;
    using matrix = types::matrix<number>;

  public:
    matrix X;
    vector Y;

    dataset(std::size_t num_features, std::size_t num_samples, std::istream &input)
        : Y(vector(num_samples)),
          X(matrix(num_samples, num_features)) {

        std::map<string, number> classes;
        int class_id = 0;

        for (std::size_t i; i < num_samples; i++) {
            auto sample = X[i];
            for (std::size_t j; j < num_features; j++) {
                // TODO: use sane file reading
                input >> a;
                sample.set(j, a);
            }
            string class_name;
            input >> class_name;
            if (classes.find(class_name) == classes.end()) {
                classes[class_name] = class_id++;
            }
            Y.set(i, classes[class_name]);
        }
    }
    void printX() {
        for (vector x : X) {
            for (auto v : x) {
                cout << v;
            }
            cout << endl;
        }
    }
};

#endif
