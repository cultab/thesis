#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPOCHS 10000

// #define matrix_index(_index_mat, _index_row, _index_col) _index_mat.data[_index_mat.cols * _index_row + _index_col]
#define vec_index(_index_vec, _index) _index_vec.col[_index]

const int PRINT_AFTER  = 15;
const int PRINT_DIGITS = 1 + PRINT_AFTER + 2;

typedef double_t f64;

const f64 smol = 10e-3;

typedef struct vec {
    size_t cols;
    f64   *col;
} vec;

typedef struct matrix {
    size_t rows;
    size_t cols;
    vec   *row;
} matrix;

#ifndef TRACE
    #define printd(var)
    #define vec_print(vec)
#else
    #define printd(var)                                                                                                \
        do {                                                                                                           \
            _printd(#var " :\t%*.*lf\n", var);                                                                         \
        } while (0)
    #define vec_print(vec)                                                                                          \
        do {                                                                                                           \
            _vec_print(vec, #vec);                                                                                  \
        } while (0)

#endif

void _printd(const char *fmt, f64 var) { printf(fmt, PRINT_DIGITS, PRINT_AFTER, var); }

void _vec_print(vec v, const char *msg)
{
    puts(msg);
    for (size_t j = 0; j < v.cols; j++) {
        printf("%*.*lf ", PRINT_DIGITS, PRINT_AFTER, vec_index(v, j));
    }
    puts("");
}

f64 Linear_Kernel(vec a, vec b)
{
    f64 res = 0;
    for (size_t k = 0; k < a.cols; k++) {
        res += a.col[k] * b.col[k];
    }
    return res;
}

#define max(a, b) a > b ? a : b
#define min(a, b) a < b ? a : b

int main(int argc, char *argv[])
{
#include "../datasets/iris.c"

    f64 COST = 0.03L;

    vec alpha = (vec) { .cols = 100, .col = (f64[100]) { 0 } };

    vec w = (vec) { .cols = 4, .col = (f64[]) { 1, 1, 1, 1 } };
    for (int i = 0; i < w.cols; i++) {
        // w.col[i] = rand() % (int)COST;
    }
    f64 b = 0;

    int epoch = 0;

    f64 (*Kernel)(vec, vec) = Linear_Kernel;

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

        f64 s = y.col[i_1] * y.col[i_2];
        f64 L = 0, H = 0;

        // find low and high
        if (y.col[i_1] != y.col[i_2]) {
            L = max(0, alpha.col[i_2] - alpha.col[i_2]);
            H = min(COST, COST + alpha.col[i_2] - alpha.col[i_2]);
        } else {
            L = max(0, alpha.col[i_2] + alpha.col[i_2] - COST);
            H = min(COST, alpha.col[i_2] - alpha.col[i_2]);
        }
        printd(L);
        printd(H);
        if (L == H) {
            continue;
        }

        // second derivative (f'')
        f64 fpp = 2 * Kernel(x.row[i_1], x.row[i_2]) - Kernel(x.row[i_1], x.row[i_1]) - Kernel(x.row[i_2], x.row[i_2]);
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
                v_1 += y.col[j] * alpha.col[j] * Kernel(x.row[i_1], x.row[j]);
            }
            f64 v_2 = 0;
            for (size_t j = 0; j < alpha.cols; j++) {
                if (j == i_1 || j == i_2) {
                    continue;
                }
                v_2 += y.col[j] * alpha.col[j] * Kernel(x.row[i_2], x.row[j]);
            }
            f64 Wconst = 0;
            for (size_t i = 0; i < alpha.cols; i++) {
                if (i == i_1 || i == i_2) {
                    continue;
                }
                Wconst += alpha.col[i];
                f64 inner_sum = 0;
                for (size_t j = 0; j < alpha.cols; j++) {
                    if (j == i_1 || j == i_2) {
                        continue;
                    }
                    inner_sum += y.col[i] * y.col[j] * Kernel(x.row[i], x.row[j]) * alpha.col[i] * alpha.col[j];
                }
                Wconst -= inner_sum / 2;
            }

            f64 gamma = alpha.col[i_1] + (s * alpha.col[i_2]);
            f64 WL    = gamma - s * L + L - (Kernel(x.row[i_1], x.row[i_1]) * (gamma - s * L)) / 2
                + (Kernel(x.row[i_2], x.row[i_2]) * (gamma - s * L) * L) / 2 - y.col[i_1] * (gamma - s * L) * v_1
                + -y.col[i_2] * L * v_2 + Wconst;
            f64 WH = gamma - s * H + H - (Kernel(x.row[i_1], x.row[i_1]) * (gamma - s * H)) / 2
                + (Kernel(x.row[i_2], x.row[i_2]) * (gamma - s * H) * H) / 2 - y.col[i_1] * (gamma - s * H) * v_1
                + -y.col[i_2] * H * v_2 + Wconst;
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
                E1 += w.col[j] * x.row[i_1].col[j] + b;
            }
            for (size_t j = 0; j < x.cols; j++) {
                E2 += w.col[j] * x.row[i_2].col[j] + b;
            }
            printd(E1);
            printd(E2);

            // new a_2
            a_2 = alpha.col[i_2] - (y.col[i_2] * (E1 - E2)) / fpp;
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
        a_1 = alpha.col[i_1] + s * (alpha.col[i_2] - a_2);
        printd(a_1);

        alpha.col[i_1] = a_1;
        alpha.col[i_2] = a_2;

        // check for some KKT conditions
        f64 a_y_is_zero = 0;
        for (size_t i = 0; i < alpha.cols; i++) {
            a_y_is_zero += alpha.col[i] * y.col[i];
        }
        f64 a_is_pos = 0;
        for (size_t i = 0; i < alpha.cols; i++) {
            a_is_pos += alpha.col[i] >= 0 ? 0 : 1;
        }
        printd(a_y_is_zero == 0);
        printd(a_is_pos == 0);

        // TODO: Compute w and use it to check for a_i to be optimized next
        for (size_t k = 0; k < w.cols; k++) {
            w.col[k] = 0;
        }

        for (size_t i = 0; i < alpha.cols; i++) {
            for (size_t k = 0; k < w.cols; k++) {
                w.col[k] += alpha.col[i] * y.col[i] * x.row[i].col[k];
            }
        }
        vec_print(alpha);
        vec_print(w);

        f64 min_p = DBL_MAX;
        for (size_t i = 0; i < x.rows; i++) {
            f64 temp = 0;
            for (size_t k = 0; k < w.cols; k++) {
                if (y.col[i] != 1) { // if not positive class skip
                    continue;
                }
                temp += w.col[k] * x.row[i].col[k];
            }
            if (temp < min_p) {
                min_p = temp;
            }
        }
        f64 max_n = -DBL_MAX;
        for (size_t i = 0; i < x.rows; i++) {
            f64 temp = 0;
            for (size_t k = 0; k < w.cols; k++) {
                if (y.col[i] != -1) { // if not negative class skip
                    continue;
                }
                temp += w.col[k] * x.row[i].col[k];
            }
            if (temp > max_n) {
                max_n = temp;
            }
        }
        b = -(min_p + max_n) / 2;
        printd(b);
    }

    vec indexes = (vec) { .cols = 100, .col = (f64[]) { 1, 51, 54, 8 } };

    int correct   = 0;
    int incorrect = 0;
    int p         = 0;
    int n         = 0;
    int tp        = 0;
    int fp        = 0;
    int tn        = 0;
    int fn        = 0;
    for (int i = 0; i < indexes.cols; i++) {
        int index = i;
        f64 res   = 0;
        puts("==========");
        printf("%d\n", index);
        vec example = x.row[index];
        // vec_print(example);
        for (int j = 0; j < example.cols; j++) {
            // puts("---");
            // printd(w.col[j]);
            // printd(example.col[j]);
            f64 temp_res = w.col[j] * example.col[j] + b;
            // printd(temp_res);
            res += temp_res;
        }
        printd(res);

        if (res > 0) {
            res = 1;
            if (fabs(y.col[index] - res) < smol) {
                tp++;
            } else {
                fp++;
            }
        } else {
            res = -1;
            if (fabs(y.col[index] - res) < smol) {
                tn++;
            } else {
                fn++;
            }
        }
        printf("actual: %d predicted: %d\n", (int)y.col[index], (int)res);
    }
    f64 accuracy  = (tp + tn) / (f64)(tp + tn + fp + fn);
    f64 recall    = tp / (f64)(tp + fn);
    f64 precision = tp / (f64)(tp + fp);
    f64 f1_score  = (2 * tp) / (f64)(2 * tp + fp + fn);
    printf("acc %lf\n", accuracy);
    printf("rec %lf\n", recall);
    printf("pre %lf\n", precision);
    printf("f1  %lf\n", f1_score);
}
