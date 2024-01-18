#ifndef DATASET_H
#define DATASET_H

#include "types.hpp"

#include <map>
#include <string>
#include <vector>

using std::fscanf;
using std::printf;
using types::index;

inline std::vector<std::string> split(const char* str, const char* delim) {
    std::vector<std::string> ret;
    char* buf = new char[255];
    size_t i = 0;
    size_t buf_len = 0;
    char c;
    do {
        c = str[i++];
        if (c == *delim || c == '\0') {
            buf_len = 0;
            ret.push_back(buf);
            // printf("push_back: '%s'\n", buf);
        } else {
            buf[buf_len++] = c;
        }

    } while (c != '\0');
    delete[] buf;
    return ret;
}

template <typename number, typename integer = int>
// requires std::floating_point<number> && std::integral<integer>
class dataset {

    using vector = types::vector<integer>;
    using matrix = types::matrix<number>;

  public:
    matrix X;
    vector Y;
    int num_samples;
    int num_features;

    dataset(size_t num_features, size_t num_samples, std::FILE* input)
        : Y(vector(num_samples)),
          X(matrix(num_samples, num_features)),
          num_samples(num_samples),
          num_features(num_features) {

        this->Y.set(0);
        std::map<std::string, integer> classes;
        integer class_id = 0;

        char* buf = new char[200];

        for (size_t i = 0; i < num_samples; i++) {
            fscanf(input, "%s", buf);
            auto values = split(buf, ",");
            size_t j;
            for (j = 0; j < num_features; j++) {
                // printf("str='%s'\n", values[j].c_str());
                std::sscanf(values[j].c_str(), "%lf", &X[i][j]); // double free or corruption (!prev)
            }
            std::string class_name = values[j];
            if (classes.find(class_name) == classes.end()) {
                classes[class_name] = class_id++;
            }
            Y[i] = classes[class_name];
        }
        delete[] buf;
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
    void printX() {
        for (size_t i = 0; i < X.rows; i++) {
            for (size_t j = 0; j < X.cols; j++) {
                std::printf("%5.3f ", X[i][j]);
            }
            std::printf("class=%d\n", Y[i]);
        }
    }
};

#endif
