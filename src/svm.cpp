#include "types.hpp"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPOCHS 10000

using vector = types::vector;
using matrix = types::matrix;

f64 Linear_Kernel(vector a, vector b) {
    f64 res = 0;
    for (size_t k = 0; k < a.cols; k++) {
        res += a.data[k] * b.data[k];
    }
    return res;
}

#define max(a, b) a > b ? a : b
#define min(a, b) a < b ? a : b

int main(int argc, char *argv[]) {

    f64 COST = 0.03L;

    vector alpha = (vector){.cols = 100, .data = (f64[100]){0}};

    vector w = (vector){.cols = 4, .data = (f64[]){1, 1, 1, 1}};
    for (int i = 0; i < w.cols; i++) {
        // w.col[i] = rand() % (int)COST;
    }
    f64 b = 0;

    int epoch = 0;

    f64 (*Kernel)(vector, vector) = Linear_Kernel;

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

        f64 s = y.data[i_1] * y.data[i_2];
        f64 L = 0, H = 0;

        // find low and high
        if (y.data[i_1] != y.data[i_2]) {
            L = max(0, alpha.data[i_2] - alpha.data[i_2]);
            H = min(COST, COST + alpha.data[i_2] - alpha.data[i_2]);
        } else {
            L = max(0, alpha.data[i_2] + alpha.data[i_2] - COST);
            H = min(COST, alpha.data[i_2] - alpha.data[i_2]);
        }
        printd(L);
        printd(H);
        if (L == H) {
            continue;
        }

        // second derivative (f'')
        f64 fpp = 2 * Kernel(x.row[i_1], x.row[i_2]) -
                  Kernel(x.row[i_1], x.row[i_1]) -
                  Kernel(x.row[i_2], x.row[i_2]);
        printd(fpp);
        if (fpp > 0) {
            puts("fpp is NOT negative!");
        }

        f64 a_1 = 0, a_2 = 0;
        if (fabs(fpp) < smol) {
            puts("fpp is zero!");
            // NOTE: see eq 12.21 again for = f^{old}(x_i) ...
            f64 v_1 = 0;
            for (size_t j = 0; j < alpha.cols; j++) {
                if (j == i_1 || j == i_2) {
                    continue;
                }
                v_1 += y.data[j] * alpha.data[j] * Kernel(x.row[i_1], x.row[j]);
            }
            f64 v_2 = 0;
            for (size_t j = 0; j < alpha.cols; j++) {
                if (j == i_1 || j == i_2) {
                    continue;
                }
                v_2 += y.data[j] * alpha.data[j] * Kernel(x.row[i_2], x.row[j]);
            }
            f64 Wconst = 0;
            for (size_t i = 0; i < alpha.cols; i++) {
                if (i == i_1 || i == i_2) {
                    continue;
                }
                Wconst += alpha.data[i];
                f64 inner_sum = 0;
                for (size_t j = 0; j < alpha.cols; j++) {
                    if (j == i_1 || j == i_2) {
                        continue;
                    }
                    inner_sum += y.data[i] * y.data[j] *
                                 Kernel(x.row[i], x.row[j]) * alpha.data[i] *
                                 alpha.data[j];
                }
                Wconst -= inner_sum / 2;
            }

            f64 gamma = alpha.data[i_1] + (s * alpha.data[i_2]);
            f64 WL =
                gamma - s * L + L -
                (Kernel(x.row[i_1], x.row[i_1]) * (gamma - s * L)) / 2 +
                (Kernel(x.row[i_2], x.row[i_2]) * (gamma - s * L) * L) / 2 -
                y.data[i_1] * (gamma - s * L) * v_1 + -y.data[i_2] * L * v_2 +
                Wconst;
            f64 WH =
                gamma - s * H + H -
                (Kernel(x.row[i_1], x.row[i_1]) * (gamma - s * H)) / 2 +
                (Kernel(x.row[i_2], x.row[i_2]) * (gamma - s * H) * H) / 2 -
                y.data[i_1] * (gamma - s * H) * v_1 + -y.data[i_2] * H * v_2 +
                Wconst;
            printd(WL);
            printd(WH);

            if (WL > WH) {
                a_2 = L;
            } else {
                a_2 = H;
            }

        } else { // if fpp is negative
            // error on training examples i_1 and i_2
            f64 E1 = 0, E2 = 0;
            for (size_t j = 0; j < x.cols; j++) {
                E1 += w.data[j] * x.row[i_1].data[j] + b;
            }
            for (size_t j = 0; j < x.cols; j++) {
                E2 += w.data[j] * x.row[i_2].data[j] + b;
            }
            printd(E1);
            printd(E2);

            // new a_2
            a_2 = alpha.data[i_2] - (y.data[i_2] * (E1 - E2)) / fpp;
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
        a_1 = alpha.data[i_1] + s * (alpha.data[i_2] - a_2);
        printd(a_1);

        alpha.data[i_1] = a_1;
        alpha.data[i_2] = a_2;

        // check for some KKT conditions
        f64 a_y_is_zero = 0;
        for (size_t i = 0; i < alpha.cols; i++) {
            a_y_is_zero += alpha.data[i] * y.data[i];
        }
        f64 a_is_pos = 0;
        for (size_t i = 0; i < alpha.cols; i++) {
            a_is_pos += alpha.data[i] >= 0 ? 0 : 1;
        }
        printd(a_y_is_zero == 0);
        printd(a_is_pos == 0);

        // TODO: Compute w and use it to check for a_i to be optimized next
        for (size_t k = 0; k < w.cols; k++) {
            w.data[k] = 0;
        }

        for (size_t i = 0; i < alpha.cols; i++) {
            for (size_t k = 0; k < w.cols; k++) {
                w.data[k] += alpha.data[i] * y.data[i] * x.row[i].data[k];
            }
        }
        vec_print(alpha);
        vec_print(w);

        f64 min_p = DBL_MAX;
        for (size_t i = 0; i < x.rows; i++) {
            f64 temp = 0;
            for (size_t k = 0; k < w.cols; k++) {
                if (y.data[i] != 1) { // if not positive class skip
                    continue;
                }
                temp += w.data[k] * x.row[i].data[k];
            }
            if (temp < min_p) {
                min_p = temp;
            }
        }
        f64 max_n = -DBL_MAX;
        for (size_t i = 0; i < x.rows; i++) {
            f64 temp = 0;
            for (size_t k = 0; k < w.cols; k++) {
                if (y.data[i] != -1) { // if not negative class skip
                    continue;
                }
                temp += w.data[k] * x.row[i].data[k];
            }
            if (temp > max_n) {
                max_n = temp;
            }
        }
        b = -(min_p + max_n) / 2;
        printd(b);
    }

    vector indexes = (vector){.cols = 100, .data = (f64[]){1, 51, 54, 8}};

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
        f64 res = 0;
        puts("==========");
        printf("%d\n", index);
        vector example = x.row[index];
        // vec_print(example);
        for (int j = 0; j < example.cols; j++) {
            // puts("---");
            // printd(w.col[j]);
            // printd(example.col[j]);
            f64 temp_res = w.data[j] * example.data[j] + b;
            // printd(temp_res);
            res += temp_res;
        }
        printd(res);

        if (res > 0) {
            res = 1;
            if (fabs(y.data[index] - res) < smol) {
                tp++;
            } else {
                fp++;
            }
        } else {
            res = -1;
            if (fabs(y.data[index] - res) < smol) {
                tn++;
            } else {
                fn++;
            }
        }
        printf("actual: %d predicted: %d\n", (int)y.data[index], (int)res);
    }
    f64 accuracy = (tp + tn) / (f64)(tp + tn + fp + fn);
    f64 recall = tp / (f64)(tp + fn);
    f64 precision = tp / (f64)(tp + fp);
    f64 f1_score = (2 * tp) / (f64)(2 * tp + fp + fn);
    printf("acc %lf\n", accuracy);
    printf("rec %lf\n", recall);
    printf("pre %lf\n", precision);
    printf("f1  %lf\n", f1_score);
}
