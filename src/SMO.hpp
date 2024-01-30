#ifndef SVM_HPP
#define SVM_HPP 1

#include <iostream>
#include <cmath>

#include "SVM_common.hpp"
#include "dataset.hpp"
#include "matrix.hpp"
#include "types.hpp"
#include "vector.hpp"

#define max(a, b) a > b ? a : b
#define min(a, b) a < b ? a : b

namespace SVM {

using types::idx; // WARN: watch out for <string.h> or <cstring>
using types::Kernel;
using types::label;
using types::matrix;
using types::number;
using types::vector;

class SMO {

  public:
    matrix& x;
    vector<label> y;
    vector<number> w;
    number b;
    Kernel K;
    vector<number> a;
    vector<number> error;

    number C;
    number tol;      // KKT tolerance
    number diff_tol; // alpha diff tolerance ?

    SMO(dataset_shape& shape, matrix& _x, vector<label>& _y, hyperparams params, Kernel kernel)
        : x(_x),
          y(std::move(_y)),
          w(shape.num_features),
          b(0),
          K(kernel),
          a(shape.num_samples),
          error(shape.num_samples),
          C(params.cost),
          tol(params.tolerance),
          diff_tol(params.diff_tolerance) {

        // w.set(1); // ERROR: watch out
        w.set(types::epsilon);
        // for (idx i = 0; i < w.cols; i++) {
        //     this->w[i] = (rand() % 10) - 5 / static_cast<number>(2);
        // }

        this->a.set(0);
        // for (idx i = 0; i < a.cols; i++) {
        //     a[i] = (rand() % 10) - 5 / (f64)2;
        // }

        // initilize error
        for (idx i = 0; i < error.cols; i++) {
            this->error[i] = predict_on(i) - y[i];
        }
    }

    void train() {
        printf("Training");
        int numChanged = 0;
        int examineAll = true;
        int epochs = 0;

        // REF: https://github.com/itsikad/svm-smo/blob/main/src/smo_optimizer.py#L231
        while (numChanged > 0 || examineAll) {
            if (epochs % 100 == 0) {
                printf(".");
                std::flush(std::cout);
            }
            if (false && epochs && epochs % 1000 == 0) {
                number avg_error = 0;
                for (auto e : error) {
                    avg_error += e;
                }
                avg_error = avg_error / static_cast<number>(error.cols);

                printf("\nContinue training? [Y/n]\n");
                printf("Already trained for %d epochs.\n", epochs);
                printf("Average error on training set: %f\n", avg_error);
                printd(w);
                int c = getchar();
                if (c == 'n') {
                    puts("Quit training!");
                    break;
                }
                if (c != '\n') {
                    getchar();
                }
            }
            numChanged = 0;

            if (examineAll) {
                // puts("examine all");
                // loop i_1 over all training examples
                for (idx i2 = 0; i2 < x.rows; i2++) {
                    numChanged += examineExample(i2);
                }
            } else {
                // puts("examine some");
                // loop i_1 over examples for which alpha is nor 0 nor Cost
                for (idx i2 = 0; i2 < x.rows; i2++) {
                    if (a[i2] != 0.0 || a[i2] != C) {
                        numChanged += examineExample(i2);
                    }
                }
            }
            if (examineAll) {
                examineAll = false;
            } else if (numChanged == 0) {
                puts("None changed, so examine all!");
                examineAll = true;
            }
            // printd(w);
            // printd(b);
            epochs++;
            if (epochs >= 10000) {
                puts("Max iteration limit reached!");
                break;
            }
        }
        printf("Done!\nTrained for %d epochs.\n", epochs);
    }

    int examineExample(idx i2) {
        // printf("examine example %zu\n", i2);

        // lookup error E2 for i_2 in error cache
        number E2 = error[i2];

        number r2 = E2 * y[i2];

        // if the error is within tolerance and the a is outside of (0, C)
        // don't change anything for this i_2
        if ((r2 < -tol && a[i2] < C) || (r2 > tol && a[i2] > 0)) {
            // number of non-zero & non-C alphas
            int non_zero_non_c = 0;
            for (idx i = 0; i < a.cols; i++) {
                if (a[i] < types::epsilon || fabs(a[i] - C) < types::epsilon) {
                    continue;
                }
                non_zero_non_c++;
                if (non_zero_non_c > 1) { // no need to count them all
                    break;
                }
            }
            if (non_zero_non_c > 1) {
                idx i1 = second_choice_heuristic(E2);
                if (takeStep(i1, i2) == 1) {
                    return 1;
                }
            }

            // TODO: cuda friendly random function
            // in the following 2 scopes
            // iters makes sure we go over all i_1
            // i_1 is the current i_1, starting from a random one, increasing until starting_i_1 - 1
            // i_1 wraps around if > a.cols

            // loop i_1 over all non-zero non-C a, starting at random point
            {

                idx iters = 0;
                idx i1 = 0;
                do {
                    i1 = static_cast<idx>(rand()) % a.cols;
                } while (i1 == i2);

                do {
                    if (fabs(a[i1]) < types::epsilon || fabs(a[i1] - C) < types::epsilon) {
                        continue;
                    }
                    if (takeStep(i1, i2) == 1) {
                        return 1;
                    }
                } while (i1 = (i1 + 1) % a.cols, iters++, iters < a.cols);
                // printf("%d\n", iters);
                // printf("%d\n", i1);
                // exit(0);
            }

            {
                idx iters = 0;
                idx i1 = 0;
                do {
                    i1 = static_cast<idx>(rand()) % a.cols;
                } while (i1 == i2);

                do {
                    if (takeStep(i1, i2) == 1) {
                        return 1;
                    }
                } while (i1 = (i1 + 1) % a.cols, iters++, iters < a.cols);
            }
        }

        // printf("%zu\n", i2);
        // puts("    within tol");
        // // if ((r2 < -tol && a[i2] < C) || (r2 > tol && a[i2] > 0)) {
        // printd(r2);
        // printd(-a[i2]);
        // printd(tol);
        // printd(C);
        // // printd(a);
        // printd(w);
        return 0;
    }

    int takeStep(idx i1, idx i2) {
        // getchar();
        // printf("    takeStep %zu %zu\n", i1, i2);
        number sign = y[i1] * y[i2];
        number L = 0, H = 0;

        // find low and high
        if (y[i1] != y[i2]) {
            L = max(0, a[i2] - a[i1]);
            H = min(C, C + a[i2] - a[i1]);
        } else {
            // printf("L = max(0, %f)\n", a[i1] + a[i2] - C);
            // printf("H = min(%f, %f)\n", C, a[i1] + a[i2]);
            L = max(0, a[i1] + a[i2] - C);
            H = min(C, a[i1] + a[i2]);
        }
        // printf("      ");
        // printd(L);
        // printf("      ");
        // printd(H);
        if (fabs(H - L) < types::epsilon) {
            // puts("      Low equals High");
            return 0;
        }

        // second derivative (f'')
        number eta = 2 * K(x[i1], x[i2]) - K(x[i1], x[i1]) - K(x[i2], x[i2]);

        // printd(eta);
        number a_1 = 0, a_2 = 0;
        if (eta < 0) { // if ("under usual circumstances") eta is negative
            // puts("      by error");
            // error on training examples i_1 and i_2
            number E1 = error[i1];
            number E2 = error[i2];
            // printd(E1);
            // printd(E2);

            // new a_2
            a_2 = a[i2] - (y[i2] * (E1 - E2)) / eta + types::epsilon;
            // printd(a_2);

            // clip a_2
            if (a_2 > H) {
                a_2 = H;
            } else if (a_2 < L) {
                a_2 = L;
            }
            // printd(a_2);
        } else {
            // TODO: eq 12.21 again for = f^{old}(x_i) ...
            // puts("      by objective eval");
            // puts("      skipping..");
            return 0;
            auto WL = eval_objective_func_at(i1, i2, L);
            auto WH = eval_objective_func_at(i1, i2, H);
            // printd(WL);
            // printd(WH);

            if (WL > WH) {
                a_2 = WL;
            } else {
                a_2 = WH;
            }
        }
        a_1 = a[i1] + sign * (a[i2] - a_2);

        // if the difference is small, don't bother
        if (fabs(a[i1] - a_1) < diff_tol) {
            return 0;
        }

        // puts("      changed\n");
        // printd(a[i1]);
        // printd(a_1);

        a[i1] = a_1;
        a[i2] = a_2;
        // printf("      ");
        // printd(a_1);
        // printf("      ");
        // printd(a_2);

        // printd(a);
        compute_w();
        b = compute_b();

        error[i1] = predict_on(i1) - y[i1];
        error[i2] = predict_on(i2) - y[i2];

        // printd(w);
        // printd(b);
        return 1;
    }

    idx second_choice_heuristic(number E2) {
        // Once a first Lagrange multiplier is chosen, SMO chooses the second Lagrange
        // multiplier to maximize the size of the step taken during joint optimization.
        // Evaluating the kernel function k is time consuming, so SMO approximates the step size
        // by the absolute value of the numerator in equation (12.6): |E1 - E2|.
        // SMO keeps a cached error value E for every non-bound example in the training set and then
        // chooses an error to approximately maximize the step size. If E1 is positive, SMO
        // chooses an example with minimum error E2. If E1 is negative, SMO chooses an
        // example with maximum error E2
        idx i1 = 0;
        number E1 = error[0];

        number min = E1;
        number max = E1;
        for (idx i = 1; i < a.cols; i++) {
            if (E2 >= 0) {
                // return idx of minimum error
                E1 = error[i];
                if (E1 < min) {
                    min = E1;
                    i1 = i;
                }
            } else {
                // return idx of maximum error
                E1 = error[i];
                if (E1 > max) {
                    max = E1;
                    i1 = i;
                } else {
                }
            }
        }

        // printf("second_choice_heuristic %zu\n", i1);
        return i1;
    }

    number predict_on(idx i) {
        // printf("      predict on %zu\n", i);
        return K(w, x[i]) + b;
    }

    number predict(vector<number>& sample) {
        // printf("      predict on %zu\n", i);
        return K(w, sample) + b;
    }

    void compute_w() {
        w.set(0);
        for (idx k = 0; k < w.cols; k++) {
            for (idx i = 0; i < a.cols; i++) {
                w[k] += a[i] * y[i] * x[i][k];
            }
        }
    }

    number compute_b() {
        number min_pos = DBL_MAX;
        number max_neg = -DBL_MAX;
        for (idx i = 0; i < x.rows; i++) {
            number tmp_p = 0;
            number tmp_n = 0;
            for (idx k = 0; k < w.cols; k++) {
                if (y[i] == 1) { // if positive class label
                    tmp_p += w[k] * x[i][k];
                } else { //    else if negative class label
                    tmp_n += w[k] * x[i][k];
                }
            }
            if (tmp_p < min_pos) {
                min_pos = tmp_p;
            }
            if (tmp_n > max_neg) {
                max_neg = tmp_n;
            }
        }

        // halfway between
        return -(min_pos + max_neg) / 2;
    }

    // evaluates the objective function at a point a2
    // W(a1, a2) = a1 + a2
    //           - 1/2 * K11 * a1^2
    //           - 1/2 * K22 * a2^2
    //           - sign* K12 * a1 *a2
    //           - y1 * a1 * v1
    //           - y2 * a2 * v2
    //           + Wconst
    // without loss of generality let the 2 multipliers be a1 and a2
    number eval_objective_func_at(idx i1, idx i2, number a2) {
        // v_i = \Sum_{j=3}^l y_j a_j^{old} K_ij
        number v_1 = 0;
        for (idx j = 0; j < a.cols; j++) {
            if (j == i1 || j == i2) { // skip i1 and i2
                continue;
            }
            v_1 += y[j] * a[j] * K(x[i1], x[j]);
        }
        number v_2 = 0;
        for (idx j = 0; j < a.cols; j++) {
            if (j == i1 || j == i2) { // skip i1 and i2
                continue;
            }
            v_2 += y[j] * a[j] * K(x[i2], x[j]);
        }

        // constant part of objective function
        // W(a) = \Sum_{i=3}^l a_i
        //      - \frac{1}{2} \Sum_{i=3}{l}\Sum_{j=3}{l} y_i y_j k(x_i, x_j) a_i a_j
        // IDEA: cache it?
        number Wconst = 0;
        for (idx i = 0; i < a.cols; i++) {
            // \Sum_{i=3}^n a_i
            if (i == i1 || i == i2) { // skip i1 and i2
                continue;
            }
            Wconst += a[i];
            // \Sum_{j=3}{l} y_i y_j k(x_i, x_j) a_i a_j
            number inner_sum = 0;
            for (idx j = 0; j < a.cols; j++) {
                if (j == i1 || j == i2) { // skip i1 and i2
                    continue;
                }
                inner_sum += y[i] * y[j] * K(x[i], x[j]) * a[i] * a[j];
            }
            Wconst -= inner_sum / 2;
        }

        // sign
        number s = y[i1] * y[i2];
        // \gamma
        number g = a[i1] + (s * a[i2]);
        // clang-format off
    return g - (s * a2) + a2
        - (K(x[i1], x[i1]) * (g - s * a2))
        / 2
        - (K(x[i2], x[i2]) * a2)
        / 2
        - s * K(i1, i2) * (g - s * a2) * a2
        - y[i1] * (g - s * a2) * v_1
        - y[i2] * a2 * v_2
        + Wconst;
        // clang-format on
    }

    void test() {
        // Testing:
        // vector indexes = (vector){.cols = 100, = (fp[]){1, 51, 54, 8}};
        vector<idx> indexes(150);
        indexes.set(1);
        indexes.mutate([](int s) -> int { return (std::rand() + s) % 150; });

        int correct = 0;
        int wrong = 0;
        for (idx sample_index = 0; sample_index < indexes.cols; sample_index++) {
            // idx i = indexes[sample_index];
            idx i = sample_index;
            number res = 0;
            puts("==========");
            printf("%zu\n", i);
            auto example = x[i];
            // printd(example);
            // for (idx j = 0; j < example.cols; j++) {
            //     res += w[j] * example[j] + b;
            // }
            res = this->predict(example);

            if (res * y[i] > 0) {
                correct++;
            } else {
                wrong++;
            }
            printf("actual: %d predicted: %f\n", y[i], res);
        }
        printd(correct);
        printd(wrong);
        number accuracy = (correct) / static_cast<number>(correct + wrong);
        printf("acc %lf\n", accuracy);

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
