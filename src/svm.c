#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define EPOCHS 1000

// #define matrix_index(_index_mat, _index_row, _index_col) _index_mat.data[_index_mat.cols * _index_row + _index_col]
#define vector_index(_index_vec, _index) _index_vec.col[_index]

const int PRINT_AFTER  = 100;
const int PRINT_DIGITS = 1 + PRINT_AFTER + 2;

typedef double_t f64;

typedef struct vector {
    size_t cols;
    f64   *col;
} vector;

typedef struct matrix {
    size_t  rows;
    size_t  cols;
    vector *row;
} matrix;

// clang-format off
vector vector_new(size_t cols)
{
    return (vector) {
        .cols = cols,
        .col = malloc(cols * sizeof(f64))
    };
}

vector vector_new_like(vector v)
{
    return (vector) {
        .cols = v.cols,
        .col = malloc(v.cols * sizeof(f64))
    };
}
// clang-format on

vector vector_zero(vector v)
{
    for (size_t i = 0; i < v.cols; i++) {
        v.col[i] = 0.0;
    }
    return v;
}

// vector vector_mult_elementwise(vector m, vector w)
// {
//     assert(m.rows == w.rows && m.cols == w.rows);
//     vector res = vector_new_like(m);
//     for (size_t i = 0; i < m.cols * m.rows; i++) {
//         res.data[i] = m.data[i] * w.data[i];
//     }
//     return res;
// }

// vector vector_mult_scalar(vector v, f64 scalar)
// {
//     vector res = vector_new_like(v);
//     for (size_t i = 0; i < v.cols; i++) {
//         res.col[i] = v.col[i] * scalar;
//     }
//     return res;
// }

vector vector_mult_scalar(vector v, size_t i, f64 scalar)
{
    vector res = vector_new_like(v);
    for (size_t j = 0; j < v.cols; j++) {
        vector_index(res, j) = v.col[j] * scalar;
    }
    return res;
}

// vector vector_mult(vector m, vector n, vector r)
// {
//     assert(m.cols == n.rows);
//     for (size_t i = 0; i < m.rows; i++) {
//         for (size_t j = 0; j < n.cols; j++) {
//             f64 temp_sum = 0;
//             for (size_t k = 0; k < m.cols; k++) {
//                 temp_sum += vector_index(m, i, k) * vector_index(n, k, j);
//             }
//             vector_index(r, i, j) = temp_sum;
//         }
//     }
//     return r;
// }

// vector vector_transpose(vector m)
// {
//     if (m.cols == 1 || m.rows == 1) {
//
//         size_t temp = m.cols;
//         m.cols      = m.rows;
//         m.rows      = temp;
//
//         return m;
//     } else {
//         assert("vector transpose not implemented for n!=m");
//         return (vector) {};
//     }
// }

#define vector_print(vec)                                                                                              \
    do {                                                                                                               \
        _vector_print(vec, #vec);                                                                                      \
    } while (0)

void _vector_print(vector v, const char *msg)
{
    puts(msg);
    for (size_t j = 0; j < v.cols; j++) {
        printf("%*.*lf ", PRINT_DIGITS, PRINT_AFTER, vector_index(v, j));
    }
    puts("");
}

#define printd(var)                                                                                                    \
    do {                                                                                                               \
        _printd(#var " :\t%*.*lf\n", var);                                                                             \
    } while (0)

void _printd(const char *fmt, f64 var) { printf(fmt, PRINT_DIGITS, PRINT_AFTER, var); }

f64 Linear_Kernel(vector a, vector b)
{
    f64 res = 0;
    for (size_t k = 0; k < a.cols; k++) {
        res += a.col[k] * b.col[k];
    }
    return res;
}

f64 eval_poly(f64 x, f64 a, f64 b, f64 c)
{
    //
    // printf("\t%f*%f^2 + %f*%f + %f = %f\n", a, x, b, x, c, a * pow(x, 2) + b * x + c);
    return a * pow(x, 2) + b * x + c;
}

#define max(a, b) a > b ? a : b
#define min(a, b) a < b ? a : b

int main(int argc, char *argv[])
{

    // clang-format off
    matrix x = (matrix) {
        .rows = 10,
        .cols = 2,
        .row = (vector[]) {
            {.cols = 2, .col = (f64[]) {1, 1}},
            {.cols = 2, .col = (f64[]) {2, 1}},
            {.cols = 2, .col = (f64[]) {4, 1}},
            {.cols = 2, .col = (f64[]) {8, 1}},
            {.cols = 2, .col = (f64[]) {9, 1}},
            {.cols = 2, .col = (f64[]) {-1, -1}},
            {.cols = 2, .col = (f64[]) {-2, -1}},
            {.cols = 2, .col = (f64[]) {-4, -1}},
            {.cols = 2, .col = (f64[]) {-8, -1}},
            {.cols = 2, .col = (f64[]) {-9, -1}},
        }
    };
    vector y = (vector) {
        .cols = 10,
        .col = (f64[]) {
             1,
             1,
             1,
             1,
             1,
            -1,
            -1,
            -1,
            -1,
            -1,
        }
    };
    vector alpha = (vector) {
        .cols = 10,
        .col = (f64[]) { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };

    vector w = (vector) {
        .cols = 2,
        .col = (f64[]) {
            1, 1
        }
    };

    f64 b = 0;
    // clang-format on
    int epoch = 0;

    while (epoch < EPOCHS) {
        int i_1 = rand() % alpha.cols; // epoch % alpha.cols;
        int i_2 = 0; //(epoch + 1) % alpha.cols;
        do {
            i_2 = rand() % alpha.cols;
        } while (i_2 == i_1);
        f64 COST = 3.0L;

        if (epoch > 0) {
            char c = getchar();
            if (c == 'q') {
                break;
            }
        }
        puts("==============");
        printf("EPOCH %d\n", epoch);
        epoch++;
        printd(i_1);
        printd(i_2);
        vector_print(y);

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

        // second derivative (f'')
        f64 fpp = 2 * Linear_Kernel(x.row[i_1], x.row[i_2]) - Linear_Kernel(x.row[i_1], x.row[i_1])
            - Linear_Kernel(x.row[i_2], x.row[i_2]);
        printd(fpp);
        if (fpp > 0) {
            puts("fpp is NOT negative!");
        }

        f64 a_1 = 0, a_2 = 0;
        if (fabs(fpp) < 0.00001) {
            puts("fpp is zero!");
            // NOTE: see eq 12.21 again for = f^{old}(x_i) ...
            f64 v_1 = 0;
            for (size_t j = 0; j < alpha.cols; j++) {
                if (j == i_1 || j == i_2) {
                    continue;
                }
                v_1 += y.col[j] * alpha.col[j] * Linear_Kernel(x.row[i_1], x.row[j]);
            }
            f64 v_2 = 0;
            for (size_t j = 0; j < alpha.cols; j++) {
                if (j == i_1 || j == i_2) {
                    continue;
                }
                v_2 += y.col[j] * alpha.col[j] * Linear_Kernel(x.row[i_2], x.row[j]);
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
                    inner_sum += y.col[i] * y.col[j] * Linear_Kernel(x.row[i], x.row[j]) * alpha.col[i] * alpha.col[j];
                }
                Wconst -= inner_sum / 2;
            }

            f64 gamma = alpha.col[i_1] + (s * alpha.col[i_2]);
            f64 WL    = gamma - s * L + L - (Linear_Kernel(x.row[i_1], x.row[i_1]) * (gamma - s * L)) / 2
                + (Linear_Kernel(x.row[i_2], x.row[i_2]) * (gamma - s * L) * L) / 2 - y.col[i_1] * (gamma - s * L) * v_1
                + -y.col[i_2] * L * v_2 + Wconst;
            f64 WH = gamma - s * H + H - (Linear_Kernel(x.row[i_1], x.row[i_1]) * (gamma - s * H)) / 2
                + (Linear_Kernel(x.row[i_2], x.row[i_2]) * (gamma - s * H) * H) / 2 - y.col[i_1] * (gamma - s * H) * v_1
                + -y.col[i_2] * H * v_2 + Wconst;
            printd(WL);
            printd(WH);

            if (WL > WH) {
                a_2 = L;
            } else {
                a_2 = H;
            }

        } else { // if fpp is negative
            puts("else");

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
        vector_print(alpha);
        vector_print(w);

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

    vector example = (vector) { .cols = 2, .col = (f64[]) { 1,  1 } };
    f64    res     = 0;
    for (int i = 0; i < example.cols; i++) {
        printd(w.col[i]);
        printd(example.col[i]);
        f64 temp_res = w.col[i] * example.col[i] + b;
        printd(temp_res);
        res += temp_res;
    }
    printd(res);
    if (res < 0) {
        puts("class -1");
    } else {
        puts("class  1");
    }
}
