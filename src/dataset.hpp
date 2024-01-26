#include <cstdlib>
#include <sstream>
#ifndef DATASET_H
#define DATASET_H 1

#include "types.hpp"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "matrix.hpp"
#include "vector.hpp"

using types::idx;
using types::label;

// REF: https://stackoverflow.com/a/46931770/13525363
inline std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(item);
    }

    return result;
}

class dataset {

    using vector = types::vector<label>;
    using matrix = types::matrix;

  public:
    matrix X;
    vector Y;
    size_t num_samples;
    size_t num_features;
    label num_classes;
    std::map<label, std::string> classes;

    dataset(size_t features, size_t samples, std::FILE* input, char delim, bool header = false)
        : X(matrix(samples, features)),
          Y(vector(samples)),
          num_samples(samples),
          num_features(features) {

        this->Y.set(0);
        std::map<std::string, label> class_to_label;
        label class_id = 0;

        char* buf = new char[200];

        puts("HERE");
        // discard header
        if (header) {
            puts("header");
            // FIX: fgets ?
            fscanf(input, "%[^\n]\n", buf);
            printf("|%s|\n", buf);
        }

        for (size_t i = 0; i < samples; i++) {
            fscanf(input, "%s", buf);
            auto values = split(buf, delim);
            size_t j;
            for (j = 0; j < features; j++) {
                printf("str='%s'\n", values[j].c_str());
                sscanf(values[j].c_str(), "%lf", &X[i][j]); // double free or corruption (!prev)
            }
            std::string class_name = values[j];
            // puts(class_name.c_str());
            if (class_to_label.find(class_name) == class_to_label.end()) {
                class_to_label[class_name] = class_id++;
            }
            Y[i] = class_to_label[class_name];
        }
        delete[] buf;
        num_classes = class_id;
        printf("num_classes=%d\n", num_classes);

        for (auto i : class_to_label) {
            classes[i.second] = i.first;
        }
    }

    ~dataset() {
        // printf("~dataset\n");
    }

    vector& getY() {
        return this->Y;
    }
    matrix& getX() {
        return this->X;
    }
};

#endif
