#include <math.h>
#include "util.h"

static inline double pois_pmf(double lambda, double x) {
    return exp(x * log(lambda) - lambda - lgamma(x + 1));
}

static double gaussian_cdf(double x) {
    if (x > 0.0) {
        if (x > 45.0) {
            return 1.0;
        } else {
            double temp = (((((x * 5.383e-6 + 4.88906e-5) * x + 3.80036e-5) * x +
                  .0032776263) * x + .0211410061) * x + .049867347) * x + 1.;
            temp *= temp;
            temp *= temp;
            temp *= temp;
            return (1. - .5 / (temp * temp));
        }
    } else {
        if (x < -45.0) {
            return 0.0;
        } else {
            x = -x;
            return (0.50 / pow((1.0 + (0.0498673470 +
                    (0.0211410061 + (0.0032776263 + (0.0000380036 +
                    (0.0000488906 + 0.0000053830 * x) * x) * x) * x) * x) *
                    x), 16));
        }
    }
}

static inline double norm_cdf(double mu, double sigma, double x) {
    return gaussian_cdf((x - mu) / sigma);
}

double pois_prob(double* par, double x) {
    return pois_pmf(par[0], x);
}

double norm_prob(double* par, double x) {
    return par[2] ? norm_cdf(par[0], par[1], x + 0.5) -
                    norm_cdf(par[0], par[1], x - 0.5) :
                    norm_cdf(par[0], par[1], 0.5);
}
