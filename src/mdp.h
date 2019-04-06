#ifndef _MDP_H_
#define _MDP_H_

double evaluate_policy(double (*func)(double*, double), double* par, int s,
                       int S, double price, double alpha);
int find_policy(double (*func)(double*, double), double* par, int S,
                double price, double alpha);
int find_all_policies(const char* ifn, const char* ofn, double alpha);
#endif
