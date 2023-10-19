#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define matrix_index(_index_mat, _index_row, _index_col) _index_mat.data[_index_mat.cols * _index_row + _index_col]
#define vector_index(_index_vec, _index) _index_vec.data[_index]

typedef double_t f64;

typedef struct matrix {
    size_t cols;
    size_t rows;
    f64   *data;
} matrix;

// clang-format off
matrix matrix_new(size_t rows, size_t cols)
{
    return (matrix) {
        .rows = rows,
        .cols = cols,
        .data = malloc(rows * cols * sizeof(f64))
    };
}

matrix matrix_new_like(matrix m)
{
    return (matrix) {
        .rows = m.rows,
        .cols = m.cols,
        .data = malloc(m.cols * m.rows * sizeof(f64))
    };
}
// clang-format on

matrix matrix_zero(matrix m)
{
    for (size_t i = 0; i < m.cols * m.rows; i++) {
        m.data[i] = 0.0;
    }
    return m;
}

matrix matrix_mult_elementwise(matrix m, matrix w)
{
    assert(m.rows == w.rows && m.cols == w.rows);
    matrix res = matrix_new_like(m);
    for (size_t i = 0; i < m.cols * m.rows; i++) {
        res.data[i] = m.data[i] * w.data[i];
    }
    return res;
}

matrix matrix_mult(matrix m, matrix n, matrix r)
{
    assert(m.cols == n.rows);
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < n.cols; j++) {
            f64 temp_sum = 0;
            for (size_t k = 0; k < m.cols; k++) {
                temp_sum += matrix_index(m, i, k) * matrix_index(n, k, j);
            }
            matrix_index(r, i, j) = temp_sum;
        }
    }
    return r;
}

matrix matrix_transpose(matrix m)
{
    if (m.cols == 1 || m.rows == 1) {

        size_t temp = m.cols;
        m.cols      = m.rows;
        m.rows      = temp;

        return m;
    } else {
        assert("Matrix transpose not implemented for n!=m");
        return (matrix) {};
    }
}

void matrix_print(matrix m, const char *msg)
{
    puts(msg);
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            printf("%6.2lf ", matrix_index(m, i, j));
        }
        puts("");
    }
    puts("");
}

f64 Kernel(matrix x, size_t a, size_t b)
{
    f64  dot = 0;
    f64 *x_i = x.data + a * x.cols;
    f64 *x_j = x.data + b * x.cols;
    for (size_t k = 0; k < x.cols; k++) {
        dot += x_i[k] * x_j[k];
    }
    return dot;
}

int main(int argc, char *argv[])
{
    // // clang-format off
    // matrix one = (matrix)
    // {
    //     .rows = 1, .cols = 3, .data = (f64[]) { 1, 2, 3 }
    // };
    // matrix two = (matrix)
    // {
    //     .rows = 3, .cols = 4, .data = (f64[]) { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }
    // };
    // // clang-format on
    // matrix res = matrix_new_like(one);
    // matrix_print(one, "one");
    // matrix_print(two, "two");
    // res = matrix_mult(one, two, res);
    // matrix_print(res, "res");

    // clang-format off
    matrix x = (matrix) {
        .rows = 3,
        .cols = 5,
        .data = (f64[]) {
              1,   1,   2,   3,   8,
             10,  10,  20,   9,  15,
            100, 115, 200, 130, 128,
        }
    };
    matrix y = (matrix) {
        .rows = 3,
        .cols = 1,
        .data = (f64[]) {
            -1,
             1,
             1
        }
    };
    matrix alpha = (matrix) {
        .rows = 1,
        .cols = 3,
        .data = (f64[]) {
            0, 0, 0
        }
    };
    // clang-format on

    int epoch = 0;

    while (epoch < 100) {
        int s = 1;
        int r = 0;
        f64 alpha_r;
        f64 alpha_s;
        // f64 rho;
        f64 zeta;
        f64 coef_a = 0;
        f64 coef_b = 0;
        f64 coef_c = 0;

        printf("%5.2lf", coef_a);
        printf("%5.2lf", coef_b);
        printf("%5.2lf", coef_c);

        // find zeta where z = - Σi∉{r,s} a_i*y_i
        for (size_t i = 0; i < alpha.cols; i++) {
            if (i == r || i == s) { // skip a_s and a_r
                continue;
            }
            zeta += alpha.data[i] * y.data[i];
        }

        alpha_r = alpha.data[r];
        alpha_s = zeta - alpha_r * y.data[r];

        for (size_t i = 0; i < alpha.cols; i++) {
            if (i == s) { // replace a_s with y_s^-1(z - a_r*y_r)
                coef_b += alpha_s;
            } else if (i == r) {
                coef_a += 1;
            } else {
                coef_b += alpha.data[i];
            }
        }

        for (size_t i = 0; i < alpha.cols; i++) {
            for (size_t j = 0; j < alpha.cols; j++) {
                if (i == r) { // coeficient of a_r -> b or a_r^2 -> a
                    if (j == r) { // coeficient of a_r^2 -> a
                        coef_a += y.data[i] * y.data[j] * Kernel(x, i, j);
                    } else { // coeficient of a_r -> b
                        if (j == s) {
                            coef_b += alpha_s * y.data[i] * y.data[j] * Kernel(x, i, j);
                        } else {
                            coef_b += alpha.data[j] * y.data[i] * y.data[j] * Kernel(x, i, j);
                        }
                    }
                } else { // coeficient of a_r^0 -> c
                    coef_c += alpha.data[i] * alpha.data[j] * y.data[i] * y.data[j] * Kernel(x, i, j);
                }
            }

        }

        // TODO: now find L and H and where y at f'(x) = 0

        //
        // for (size_t i = 0; i < alpha.cols; i++) {
        //     for (size_t j = 0; j < alpha.cols; j++) { }
        //
        //     // replace x[i] with kernel(x[i])
        //
        //     // f64 z = sum alpha[i] * y[i] for i not in {r,s}
        //     // look for a_s where:
        //     // f64 a_s = (z - alpha[r] * y[r]) / y[s]
        //
        //     // choose r at random for now ?
        //
        //     // chose alpha[r] to maximise a*alpha[r]^2+b*alpha[r] + c
        //     // s.t.
        //     // if y[r] == y[s] max(0, -y[r]*C-z) <= alpha[r] <= min(C, y[r]*z)
        //     // if y[r] != y[s] max(0, y[r]*z) <= alpha[r] <= min(C, -y[r]*C-z)
        //     //
        //     // where a b c are the fixed alpha[i]s'?
        //     // alpha[r] = -b/2a or
        //     //
        //     // a = (sum all alpha[i] but alpha[r])
        //     // b = sum all alpha[i]
        //     // c =
        //     // sum all alpha[i]
        //     // - 1/2
        //     // ar*ar    a1*ar ar*a1 a1*ar
        //     //        ar*(a1+a1+a1)
        //     //
        //     // a = 1 ?
        //     // b = 2 * [ (sum alpha[i]) - alpha[r] ] + 1
        //     // c = sum alpha[i]*alpha[j] skipping alpha[r] ?
        //
        //     // C = rho[i] + alpha[i] (choose rho so that alpha[i] <= C)
        //     // so
        //     // 0 < alpha[i] < C
        //
        //     // alpha[i] != 0 for support vectors
        //     // w is linear combination of support vectors
        //
        //     // w = sum alpha[i]y[i]x[i]
        //     // b = -1/2 (min      sum    a_j*y_j*K(xi,xj) + max but for i:y_i=-1)
        //     //           i:y_i=1  a_j!=0       i:y_i=-1
        //
        //     // predict by sign(w^T*x+b)
        //     // or sign(sum ai yi K(xi, x) +b)
        //     //         ai!=0
        //     return 0;
    }
}
