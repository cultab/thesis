
//{ // QP
// // find zeta where z = - Σi∉{r,s} a_i*y_i
// for (size_t i = 0; i < alpha.cols; i++) {
//     if (i == r || i == s) { // skip a_s and a_r
//         continue;
//     }
//     // printf("zeta -= %lf ", alpha.data[i] * y.data[i]);
//     zeta -= alpha.data[i] * y.data[i];
// }
// // zeta = - zeta;
// printd(zeta);
//
// // first Σ
// for (size_t i = 0; i < alpha.cols; i++) {
//     if (i == r) {
//         coef_b += 1;
//     } else if (i == s) {
//         coef_b += -(y.data[r] / y.data[s]);
//         coef_c += zeta / y.data[s];
//     } else {
//         coef_c += alpha.data[i];
//     }
// }
//
//
// f64 coef_a_temp = 0, coef_b_temp = 0, coef_c_temp = 0;
// // second Σ
// for (size_t i = 0; i < alpha.cols; i++) {
//     for (size_t j = 0; j < alpha.cols; j++) {
//         f64 yyK = y.data[i] * y.data[j] * Kernel(x, i, j);
//         if (i == r && j == r) { // a
//             coef_a_temp += yyK;
//         } else if (i == r || j == r) {
//             if (i == s || j == s) { // a again
//                 coef_b_temp += -(y.data[r] / y.data[s]) * yyK;
//                 coef_c_temp += zeta / y.data[s] * yyK;
//             } else { // b
//                 coef_b_temp += ((zeta - y.data[r]) / y.data[s]) * yyK;
//             }
//         } else if (i == s && j == s) {
//             coef_b_temp
//                 += -(y.data[r] / y.data[s]) * -(y.data[r] / y.data[s]) * yyK;
//             coef_c_temp += (zeta / y.data[s]) * (-zeta / y.data[s]) * yyK;
//
//         } else if (j == s) { // b again
//             coef_b_temp += alpha.data[i] * -(y.data[r] / y.data[s]) * yyK;
//             coef_c_temp += alpha.data[i] * (zeta / y.data[s]) * yyK;
//         } else if (i == s) { // b again
//             coef_b_temp += -(y.data[r] / y.data[s]) * alpha.data[j] * yyK;
//             coef_c_temp += (zeta / y.data[s]) * alpha.data[j] * yyK;
//         } else { // c finally
//             coef_c_temp += alpha.data[i] * alpha.data[j] * yyK;
//         }
//     }
// }
//
// // take -1/2 * ... into account
// coef_a += -coef_a_temp / 2;
// coef_b += -coef_b_temp / 2;
// coef_c += -coef_c_temp / 2;
// printd(coef_a);
// printd(coef_b);
// printd(coef_c);
//
// // TODO: now find L and H and where y at f'(x) = 0
//
// f64 crit = -coef_b / (2 * coef_a);
//
// // find low and high endpoint
// f64 L, H;
// if (y.data[r] == y.data[s]) {
//     puts("y_r == y_s");
//     // printf("L = -(%fll * (%fll * %fll - %fll))\n", y.data[r], y.data[s], COST, zeta);
//     L = fmax(0, -y.data[r] * (y.data[s] * COST - zeta));
//     H = fmin(COST, y.data[r] * zeta);
// } else {
//     puts("y_r != y_s");
//     L = fmax(0, y.data[r] * zeta);
//     H = fmin(COST, -y.data[r] * (y.data[s] * COST - zeta));
// }
//
// printd(L);
// printd(crit);
// printd(H);
// f64 L_val    = eval_poly(L, coef_a, coef_b, coef_c);
// f64 crit_val = eval_poly(crit, coef_a, coef_b, coef_c);
// f64 H_val    = eval_poly(H, coef_a, coef_b, coef_c);
// printd(L_val);
// printd(crit_val);
// printd(H_val);
//
// if (fabs(coef_a) < 0.0001) { // if coef_a = 0 only eval L and H
//     printf("\ta == 0\n");
//     // f64 L_val = eval_poly(L, coef_a, coef_b, coef_c);
//     // f64 H_val = eval_poly(H, coef_a, coef_b, coef_c);
//     if (L_val > H_val) {
//         alpha_r_new = L; // take the L
//     } else {
//         alpha_r_new = H;
//     }
// } else { // if coef_b != 0, eval all
//     puts("\ta != 0");
//     if (L <= crit && crit <= H) { // if critical point between endpoints, concider it for max
//         puts("concider crit");
//         if (crit_val >= L_val && crit_val >= H_val) { // if bigger than both == max
//             puts("\tcrit selected");
//             alpha_r_new = crit;
//         } else if (L_val > H_val) {
//             puts("\tL selected");
//             alpha_r_new = L;
//         } else {
//             puts("\tH selected");
//             alpha_r_new = H;
//         }
//     } else { // only concider end points
//         puts("only concider endpoints");
//         if (L_val > H_val) {
//             printf("L selected\n");
//             alpha_r_new = L;
//         } else {
//             printf("H selected\n");
//             alpha_r_new = H;
//         }
//     }
// }

// vector_print(alpha);
// if (alpha_r_new == 0 && epoch <= 4) {
//     puts("cheat");
//     alpha_r_new = 1;
// }
// alpha.data[r] = alpha_r_new;
// alpha.data[s] = (zeta - alpha_r_new * y.data[r]) / y.data[s]; // NOW calculate new a_s
// printd(alpha.data[r]);
// printd(alpha.data[s]);
// vector_print(alpha);
//}
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
        //     // alpha[i] != 0 for support OLD_vect0rs
        //     // w is linear combination of support OLD_vect0rs
        //
        //     // w = sum alpha[i]y[i]x[i]
        //     // b = -1/2 (min      sum    a_j*y_j*K(xi,xj) + max but for i:y_i=-1)
        //     //           i:y_i=1  a_j!=0       i:y_i=-1
        //
        //     // predict by sign(w^T*x+b)
        //     // or sign(sum ai yi K(xi, x) +b)
        //     //         ai!=0
        //     return 0;
