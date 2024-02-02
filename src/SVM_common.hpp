#ifndef SVM_COMMON_HPP
#define SVM_COMMON_HPP

#include "types.hpp"

using types::math_t;

namespace SVM {

enum Kernel_t { POLY, RBF, LINEAR };

static const char* Kernel_t_name[] = {"POLYNOMIAL", "RADIAL BASIS FUNC", "LINEAR"};

struct hyperparams {
    math_t cost;
    math_t tolerance;
    math_t diff_tolerance;
};
} // namespace SVM
#endif // SVM_COMMON_HPP
