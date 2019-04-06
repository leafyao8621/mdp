#include <stdio.h>
#include "util.h"
#include "mdp.h"
#include "matrix.h"

int main(void) {
    double pars[3];
    // *pars = 50;
    // for (int i = 0; i < 100; i++) {
    //     printf("pois 50 %d %lf\n", i, pois_prob(pars, i));
    // }
    // pars[1] = 17;
    // for (int i = 0; i < 100; i++) {
    //     pars[2] = i;
    //     printf("norm 50 17 %d %lf\n", i, norm_prob(pars, i));
    // }
    // *pars = 10;
    // for (int i = 1; i < 21; i++) {
    //     printf("%lf\n", evaluate_policy(pois_prob, pars, i, 20, 50, 1.5));
    // }
    // printf("%d\n", find_policy(pois_prob, pars, 20, 5, 1.5));
    find_all_policies("data_to_markov.mdp", "out.csv", 2);
}
