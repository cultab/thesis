#ifndef OVA_HPP
#define OVA_HPP 1
#include "SVM_serial.hpp"
#include "dataset.hpp"

using types::idx;
using types::label;
using types::vector;

namespace SVM {

// One V All
// n models for n classes
class OVA {

    vector<SVM*> models;
    hyperparams params;

  public:
    OVA(dataset_shape& shape, matrix& x, vector<label>& y, hyperparams _params, types::Kernel kernel)
        : models(static_cast<idx>(shape.num_classes == 2 ? 1 : shape.num_classes)),
          params(_params) {

        for (label /*class_id*/ cls = 0; static_cast<idx>(cls) < models.cols; cls++) {
            // IDEA: data.y mutate so that 1 for i and -1 for others
            // copy labels vector
            vector<label> labels = y;
            // mutate as mentioned
            labels.mutate([cls](int l) -> int { return l == cls ? 1 : -1; });
            // train svm
            auto model = new SVM(shape,x, labels, this->params, kernel);
            models[static_cast<idx>(cls)] = model;
        }
    }

    void train() {
        for (idx i = 0; i < models.cols; i++) {
            printf("Training model %zu\n", i);
            models[i]->train();
            printd(models[i]->w);
        }
    }

    std::tuple<label, number> predict(vector<number> sample) {
        label cls = 0;
        number max_pred = models[static_cast<idx>(cls)]->predict(sample);
        printf("model %d: %f\n", cls, max_pred);

        for (idx i = 1; i < models.cols; i++) {
            number tmp = models[i]->predict(sample);
            printf("model %zu: %f\n", i, tmp);
            if (tmp > max_pred) {
                cls = static_cast<label>(i);
                max_pred = tmp;
            }
        }
        if (models.cols == 1) {
            cls = max_pred > 0 ? 0 : 1;
        }
        return std::tuple<label, number>(cls, max_pred);
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
            puts("==========");
            printf("%zu\n", i);
            vector<number> example = data.X[i];

            auto [pred, score] = this->predict(example);

            printf("actual: %s predicted: %s\n", data.classes[data.Y[i]].c_str(), data.classes[pred].c_str());
            printf("actual: %d predicted: %d\n", data.Y[i], pred);
            printf("score: %f\n", score);
            if (pred == data.Y[i]) {
                // puts("CORRECT");
                correct++;
            } else {
                puts("WRONG");
                incorrect++;
            }
        }
        printd(correct);
        printd(incorrect);
        number accuracy = (correct) / static_cast<number>(correct + incorrect);
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
