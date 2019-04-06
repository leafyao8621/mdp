#ifndef _MDP_H_
#define _MDP_H_

double evaluate_policy(double (*func)(double*, double), double* par, int s,
                       int S, double price, double alpha);
#endif
