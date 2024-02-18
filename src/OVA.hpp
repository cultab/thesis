#ifndef OVA_HPP
#define OVA_HPP 1
#include "SVM_common.hpp"
#include "dataset.hpp"

using types::idx;
using types::label;
using types::vector;
using types::matrix;

namespace SVM {

// One V All
// n models for n classes
template <typename SVM_IMPL>
class OVA {

    vector<SVM_IMPL*> models;
    hyperparams params;

  public:
    OVA(dataset_shape& shape, matrix<math_t>& x, vector<label>& y, hyperparams _params, Kernel_t kernel)
        : models(static_cast<idx>(shape.num_classes == 2 ? 1 : shape.num_classes)),
          params(_params) {

        for (label /*class_id*/ cls = 0; static_cast<idx>(cls) < models.cols; cls++) {
            // IDEA: data.y mutate so that 1 for i and -1 for others
            // copy labels vector
            vector<label> labels = y;
            // mutate as mentioned
            labels.mutate([cls](int l) -> int { return l == cls ? 1 : -1; });
            // train svm
            auto model = new SVM_IMPL(shape,x, labels, this->params, kernel);
            models[static_cast<idx>(cls)] = model;
        }
    }

    void train() {
        for (idx i = 0; i < models.cols; i++) {
            printf("Training model %zu\n", i);
            models[i]->train();
            models[i]->compute_w();
            printf("Done training model %zu\n", i);
            printd(models[i]->w);
        }
    }

    std::tuple<label, math_t> predict(vector<math_t> sample) {
        label cls = 0;
        math_t max_pred = models[static_cast<idx>(cls)]->predict(sample);
        printf("model %d: %f\n", cls, max_pred);

        for (idx i = 1; i < models.cols; i++) {
            math_t tmp = models[i]->predict(sample);
            printf("model %zu: %f\n", i, tmp);
            if (tmp > max_pred) {
                cls = static_cast<label>(i);
                max_pred = tmp;
            }
        }
        if (models.cols == 1) {
            cls = max_pred > 0 ? 0 : 1;
        }
        return std::tuple<label, math_t>(cls, max_pred);
    }

    ~OVA() {
        for (auto m : models) {
            delete m;
        }
    }

    void test(dataset& data) {
        int correct = 0;
        int incorrect = 0;
        for (idx i = 0; i < data.shape.num_samples; i++) {
            // if (!(i % 10 == 0)) {
            //     continue;
            // }
            vector<math_t> example = data.X[i];

            auto [pred, score] = this->predict(example);
            // puts("==========");
            // printf("%zu\n", i);
            // printf("actual: %s predicted: %s\n", data.classes[data.Y[i]].c_str(), data.classes[pred].c_str());
            // printf("actual: %d predicted: %d\n", data.Y[i], pred);
            // printf("score: %f\n", score);
            if (pred == data.Y[i]) {
                // puts("CORRECT");
                correct++;
            } else {
                // puts("WRONG");
                incorrect++;
            }
        }
        printd(correct);
        printd(incorrect);
        math_t accuracy = (correct) / static_cast<math_t>(correct + incorrect);
        printf("acc %lf\n", accuracy);

        // for (auto m : models) {
        //     m->test();
        // }

        // printf("a:   %p\n", &a);
        // printf("w:       %p\n", &w);
        // printf("indexes: %p\n", &indexes);
        // printf("x:       %p\n", &x);
        // printf("y:       %p\n", &y);
        // printf("d.x:     %p\n", &data.X);
        // printf("d.y:     %p\n", &data.Y);
        // printf("example: %p\n", &example);
    }
};

} // namespace SVM
#endif
