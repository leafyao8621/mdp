#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "matrix.h"
#include "util.h"

static double* gen_header(double (*func)(double*, double), double* par, int S) {
    double* out = vector_alloc(S + 1);
    double* j = out + S;
    double sum = 0;
    for (int i = 0; i < S; *j = func(par, i++), sum += *(j--));
    *j = 1 - sum;
    return out;
}

static double** gen_matrix(double (*func)(double*, double), double* par, int s,
                           int S) {
    double** out = matrix_calloc(S + 1, S + 1);
    double* header = gen_header(func, par, S);
    double** j = out;
    int i = 0;
    for (; i < s; i++, j++) {
        memcpy(*j, header, sizeof(double) * (S + 1));
        (*j)[S] = -1;
    }
    memcpy(out[S], header, sizeof(double) * (S + 1));
    for (i = S - 1, j = out + S - 1; i >= s; i--, j--) {
        **j = j[1][0] + j[1][1];
        memcpy(*j + 1, j[1] + 2, sizeof(double) * i);
        (*j)[S] = -1;
    }
    out[S][S] = 0;
    vector_free(header);
    return out;
}

static double* gen_reward(double** P, int s, int S, double price,
                          double alpha) {
    double* out = vector_alloc(S + 1);
    int i = 0;
    double* j = out;
    double** k = P;
    for (; i < s; i++, j++, k++) {
        *j = (i - S) * price - S * (**k) * price * alpha;
    }
    for (; i < S + 1; i++, j++, k++) {
        *j = -S * (**k) * price * alpha;
    }
    return out;
}

double evaluate_policy(double (*func)(double*, double), double* par, int s,
                       int S, double price, double alpha) {
    double** P = gen_matrix(func, par, s, S);
    double** I = create_identity(S + 1);
    double* r = gen_reward(P, s, S, price, alpha);
    double* res = vector_alloc(S + 1);
    matrix_subtract(S + 1, S + 1, I, P, P);
    matrix_invert(S + 1, P, P);
    left_multiply(S + 1, S + 1, P, r, res);
    double out = res[S];
    vector_free(r);
    matrix_free(S + 1, P);
    matrix_free(S + 1, I);
    return out;
}

int find_policy(double (*func)(double*, double), double* par, int S,
                double price, double alpha) {
    int lb, ub;
    lb = 1;
    ub = S;
    int s = 1 + S >> 1;
    double val;
    double l, r;
    for (;;) {
        val = evaluate_policy(func, par, s, S, price, alpha);
        if (s > 1 && s < S) {
            l = evaluate_policy(func, par, s - 1, S, price, alpha);
            r = evaluate_policy(func, par, s + 1, S, price, alpha);
            if (val >= l && val >= r) {
                return s;
            } else if (l > r) {
                ub = s;
                if (ub - lb > 1) {
                    s = ub - ((ub - lb) >> 1);
                } else {
                    s = lb;
                }
            } else {
                lb = s;
                if (ub - lb > 1) {
                    s = lb + ((ub - lb) >> 1);
                } else {
                    s = ub;
                }
            }
        } else if (s == 0) {
            r = evaluate_policy(func, par, s + 1, S, price, alpha);
            if (val >= r) {
                return s;
            } else {
                lb = s;
                if (ub - lb == 1) {
                    s = ub;
                } else {
                    s = lb + ((ub - lb) >> 1);
                }
            }
        } else if (s == S) {
            l = evaluate_policy(func, par, s - 1, S, price, alpha);
            if (val >= l) {
                return s;
            } else {
                ub = s;
                if (ub - lb == 1) {
                    s = lb;
                } else {
                    s = ub - ((ub - lb) >> 1);
                }
            }
        }
    }
    return -1;
}

int find_all_policies(const char* ifn, const char* ofn, double alpha) {
    FILE* fin = fopen(ifn, "r");
    FILE* fout = fopen(ofn, "w");
    char id_buf[30];
    int S;
    double price;
    char type;
    double par[2];
    int s;
    for (; !feof(fin);) {
        fscanf(fin, "%s %d %lf %c", id_buf, &S, &price, &type);
        puts(id_buf);
        fflush(stdout);
        switch (type) {
        case 'p':
        case 'P':
            scanf("%lf", par);
            fprintf(fout, "%s,%d,%d\n", id_buf, find_policy(pois_prob, par, S,
                                                            price, alpha), S);
            break;
        case 'n':
        case 'N':
            scanf("%lf,%lf", par, par + 1);
            fprintf(fout, "%s,%d,%d\n", id_buf, find_policy(norm_prob, par, S,
                                                            price, alpha), S);
            break;
        }
    }
    fclose(fin);
    fclose(fout);
    return 0;
}
