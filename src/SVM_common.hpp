#ifndef SVM_COMMON_HPP
#define SVM_COMMON_HPP

#include "types.hpp"

using types::number;

namespace SVM {

struct hyperparams {
    number cost;
    number tolerance;
    number diff_tolerance;
};
} // namespace SVM
#endif // SVM_COMMON_HPP
