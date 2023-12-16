#include "dataset.hpp"
#include "types.hpp"
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPOCHS 10000

using fl = types::f64;
using types::matrix;
using types::vector;

fl Linear_Kernel(vector<fl> a, vector<fl> b) {
    fl res = 0;
    for (size_t k = 0; k < a.cols; k++) {
        res += a[k] * b[k];
    }
    return res;
}

#define max(a, b) a > b ? a : b
#define min(a, b) a < b ? a : b

int main(int argc, char *argv[]) {

    fl COST = 0.03L;

    auto file = std::fopen("../datasets/iris", "r");
    dataset<fl> data(4, 150, file);
    vector<int> y = std::move(data.Y);
    matrix<fl> x = std::move(data.X);

    vector alpha(100);
    alpha.set(0);
    vector w(4);
    w.set(1);
    fl b = 0;

    int epoch = 0;

    fl (*Kernel)(vector<fl>, vector<fl>) = Linear_Kernel;

    while (epoch < EPOCHS) {
        int i_1 = rand() % alpha.cols;
        int i_2 = 0;
        do {
            i_2 = rand() % alpha.cols;
        } while (i_2 == i_1);

        // if (epoch > 0) {
        //     char c = getchar();
        //     if (c == 'q') {
        //         break;
        //     }
        // }

        puts("==============");
        printf("EPOCH %d\n", epoch);
        epoch++;
        printd(i_1);
        printd(i_2);
        vec_print(y);

        fl s = y[i_1] * y[i_2];
        fl L = 0, H = 0;

        // find low and high
        if (y[i_1] != y[i_2]) {
            L = max(0, alpha[i_2] - alpha[i_2]);
            H = min(COST, COST + alpha[i_2] - alpha[i_2]);
        } else {
            L = max(0, alpha[i_2] + alpha[i_2] - COST);
            H = min(COST, alpha[i_2] - alpha[i_2]);
        }
        printd(L);
        printd(H);
        if (L == H) {
            continue;
        }

        // second derivative (f'')
        fl fpp = 2 * Kernel(x[i_1], x[i_2]) - Kernel(x[i_1], x[i_1]) - Kernel(x[i_2], x[i_2]);
        printd(fpp);
        if (fpp > 0) {
            puts("fpp is NOT negative!");
        }

        fl a_1 = 0, a_2 = 0;
        if (fabs(fpp) < types::epsilon) {
            puts("fpp is zero!");
            // NOTE: see eq 12.21 again for = f^{old}(x_i) ...
            fl v_1 = 0;
            for (size_t j = 0; j < alpha.cols; j++) {
                if (j == i_1 || j == i_2) {
                    continue;
                }
                v_1 += y[j] * alpha[j] * Kernel(x[i_1], x[j]);
            }
            fl v_2 = 0;
            for (size_t j = 0; j < alpha.cols; j++) {
                if (j == i_1 || j == i_2) {
                    continue;
                }
                v_2 += y[j] * alpha[j] * Kernel(x[i_2], x[j]);
            }
            fl Wconst = 0;
            for (size_t i = 0; i < alpha.cols; i++) {
                if (i == i_1 || i == i_2) {
                    continue;
                }
                Wconst += alpha[i];
                fl inner_sum = 0;
                for (size_t j = 0; j < alpha.cols; j++) {
                    if (j == i_1 || j == i_2) {
                        continue;
                    }
                    inner_sum += y[i] * y[j] * Kernel(x[i], x[j]) * alpha[i] * alpha[j];
                }
                Wconst -= inner_sum / 2;
            }

            fl gamma = alpha[i_1] + (s * alpha[i_2]);
            fl WL = gamma - s * L + L - (Kernel(x[i_1], x[i_1]) * (gamma - s * L)) / 2 +
                    (Kernel(x[i_2], x[i_2]) * (gamma - s * L) * L) / 2 - y[i_1] * (gamma - s * L) * v_1 +
                    -y[i_2] * L * v_2 + Wconst;
            fl WH = gamma - s * H + H - (Kernel(x[i_1], x[i_1]) * (gamma - s * H)) / 2 +
                    (Kernel(x[i_2], x[i_2]) * (gamma - s * H) * H) / 2 - y[i_1] * (gamma - s * H) * v_1 +
                    -y[i_2] * H * v_2 + Wconst;
            printd(WL);
            printd(WH);

            if (WL > WH) {
                a_2 = L;
            } else {
                a_2 = H;
            }

        } else { // if fpp is negative
            // error on training examples i_1 and i_2
            fl E1 = 0, E2 = 0;
            for (size_t j = 0; j < x.cols; j++) {
                E1 += w[j] * x[i_1][j] + b;
            }
            for (size_t j = 0; j < x.cols; j++) {
                E2 += w[j] * x[i_2][j] + b;
            }
            printd(E1);
            printd(E2);

            // new a_2
            a_2 = alpha[i_2] - (y[i_2] * (E1 - E2)) / fpp;
            printd(a_2);

            // clip a_2
            if (a_2 > H) {
                a_2 = H;
            } else if (a_2 < L) {
                a_2 = L;
            }
            printd(a_2);
        }
        // numerical stability???
        // if (a_2 == 0) {
        //     a_2 = DBL_EPSILON;
        // }
        a_1 = alpha[i_1] + s * (alpha[i_2] - a_2);
        printd(a_1);

        alpha[i_1] = a_1;
        alpha[i_2] = a_2;

        // check for some KKT conditions
        fl a_y_is_zero = 0;
        for (size_t i = 0; i < alpha.cols; i++) {
            a_y_is_zero += alpha[i] * y[i];
        }
        fl a_is_pos = 0;
        for (size_t i = 0; i < alpha.cols; i++) {
            a_is_pos += alpha[i] >= 0 ? 0 : 1;
        }
        printd(a_y_is_zero == 0);
        printd(a_is_pos == 0);

        // TODO: Compute w and use it to check for a_i to be optimized next
        for (size_t k = 0; k < w.cols; k++) {
            w[k] = 0;
        }

        for (size_t i = 0; i < alpha.cols; i++) {
            for (size_t k = 0; k < w.cols; k++) {
                w[k] += alpha[i] * y[i] * x[i][k];
            }
        }
        vec_print(alpha);
        vec_print(w);

        fl min_p = DBL_MAX;
        for (size_t i = 0; i < x.rows; i++) {
            fl temp = 0;
            for (size_t k = 0; k < w.cols; k++) {
                if (y[i] != 1) { // if not positive class skip
                    continue;
                }
                temp += w[k] * x[i][k];
            }
            if (temp < min_p) {
                min_p = temp;
            }
        }
        fl max_n = -DBL_MAX;
        for (size_t i = 0; i < x.rows; i++) {
            fl temp = 0;
            for (size_t k = 0; k < w.cols; k++) {
                if (y[i] != -1) { // if not negative class skip
                    continue;
                }
                temp += w[k] * x[i][k];
            }
            if (temp > max_n) {
                max_n = temp;
            }
        }
        b = -(min_p + max_n) / 2;
        printd(b);
    }

    // vector indexes = (vector){.cols = 100, = (fp[]){1, 51, 54, 8}};
    vector<int> indexes(4);
    indexes[0] = 1;
    indexes[1] = 51;
    indexes[2] = 54;
    indexes[3] = 8;

    int correct = 0;
    int incorrect = 0;
    int p = 0;
    int n = 0;
    int tp = 0;
    int fp = 0;
    int tn = 0;
    int fn = 0;
    for (int i = 0; i < indexes.cols; i++) {
        int index = i;
        fl res = 0;
        puts("==========");
        printf("%d\n", index);
        vector example = x[index];
        // vec_print(example);
        for (int j = 0; j < example.cols; j++) {
            // puts("---");
            // printd(w.col[j]);
            // printd(example.col[j]);
            fl temp_res = w[j] * example[j] + b;
            // printd(temp_res);
            res += temp_res;
        }
        printd(res);

        if (res > 0) {
            res = 1;
            if (fabs(y[index] - res) < types::epsilon) {
                tp++;
            } else {
                fp++;
            }
        } else {
            res = -1;
            if (fabs(y[index] - res) < types::epsilon) {
                tn++;
            } else {
                fn++;
            }
        }
        printf("actual: %d predicted: %d\n", (int)y[index], (int)res);
    }
    fl accuracy = (tp + tn) / (fl)(tp + tn + fp + fn);
    fl recall = tp / (fl)(tp + fn);
    fl precision = tp / (fl)(tp + fp);
    fl f1_score = (2 * tp) / (fl)(2 * tp + fp + fn);
    printf("acc %lf\n", accuracy);
    printf("rec %lf\n", recall);
    printf("pre %lf\n", precision);
    printf("f1  %lf\n", f1_score);
}
