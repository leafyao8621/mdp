#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "matrix.h"

/**
 * allocates memory for a vector
 * @param dim dimension
 * @return    the vector
 */
double* vector_alloc(int n) {
    double* output = malloc(sizeof(double*) * n);
    if (output == NULL) {
        puts("vector alloc malloc error");
        return NULL;
    }
    return output;
}

/**
 * allocates memory for a matrix in parallel
 * @param  nr number of rows
 * @param  nc number of columns
 * @return    the matrix
 */
double** matrix_alloc(int nr, int nc) {
    double** output = malloc(sizeof(double*) * nr);
    if (output == NULL) {
        puts("matrix alloc malloc error");
        return NULL;
    }
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int inc = omp_get_num_threads();
        for (int i = id; i < nr; i += inc) {
            output[i] = malloc(sizeof(double) * nc);
            if (output[i] == NULL) {
                puts("matrix alloc malloc error");
                exit(1);
            }
        }
    }
    return output;
}

/**
 * conducts calloc for memory allocation of a matrix in parallel
 * @param  nr number of rows
 * @param  nc number of columns
 * @return    the matrix
 */
double** matrix_calloc(int nr, int nc) {
    double** output = malloc(sizeof(double*) * nr);
    if (output == NULL) {
        puts("matrix calloc malloc error");
        return NULL;
    }
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int inc = omp_get_num_threads();
        for (int i = id; i < nr; i += inc) {
            output[i] = calloc(nc, sizeof(double));
            if (output[i] == NULL) {
                puts("matrix calloc calloc error");
                exit(1);
            }
        }
    }
    return output;
}

/**
 * couducts matrix multiplication on two matrices in parallel
 * @param  nra    number of rows of the left operand matrix
 * @param  nca    number of columns of the left operand matrix/number of rows of the right operand matrix
 * @param  ncb    number of columns of the right operand matrix
 * @param  a      the left operand matrix
 * @param  b      the right operand matrix
 * @param  output the output matrix
 * @return        whether the operation is successful
 */
int matrix_multiply(int nra, int nca, int ncb, double** a, double** b, double** output) {
    if (a == NULL || b == NULL || output == NULL) {
        puts("matrix multiply NULL ptr");
        return 1;
    }
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int inc = omp_get_num_threads();
        for (int i = id; i < nra; i += inc) {
            for (int j = 0; j < ncb; j++) {
                double temp = 0;
                for (int k = 0; k < nca; k++) {
                    temp += a[i][k] * b[k][j];
                }
                output[i][j] = temp;
            }
        }
    }
    return 0;
}

/**
 * conducts matrix addition on two matrices in parallel
 * @param  nr     number of rows of the matrices
 * @param  nc     number of columns of the matrices
 * @param  a      the left operand matrix
 * @param  b      the right operand matrix
 * @param  output the output matrix
 * @return        whether the operation is successful
 */
int matrix_add(int nr, int nc, double** a, double** b, double** output) {
    if (a == NULL || b == NULL || output == NULL) {
        puts("matrix add NULL ptr");
        return 1;
    }
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int inc = omp_get_num_threads();
        for (int i = id; i < nr; i += inc) {
            for (int j = 0; j < nc; j++) {
                output[i][j] = a[i][j] + b[i][j];
            }
        }
    }
    return 0;
}

/**
 * couducts matrix subtraction on two matrices in parallel
 * @param  nr     number of rows of the matrices
 * @param  nc     number of columns of the matrices
 * @param  a      the left operand matrix
 * @param  b      the right operand matrix
 * @param  output the output matrix
 * @return        whether the operation is successful
 */
int matrix_subtract(int nr, int nc, double** a, double** b, double** output) {
    if (a == NULL || b == NULL || output == NULL) {
        puts("matrix subtract NULL ptr");
        return 1;
    }
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int inc = omp_get_num_threads();
        for (int i = id; i < nr; i += inc) {
            for (int j = 0; j < nc; j++) {
                output[i][j] = a[i][j] - b[i][j];
            }
        }
    }
    return 0;
}

/**
 * conducts vector matrix right multiplication in parallel
 * @param  da     dimension of the vector/number of rows of the matrix
 * @param  ncb    number of columns of the matrix
 * @param  a      the vector
 * @param  b      the matrix
 * @param  output the output vector
 * @return        whether the operation is successful
 */
int right_multiply(int da, int ncb, double* a, double** b, double* output) {
    if (a == NULL || b == NULL || output == NULL) {
        puts("right multiply NULL ptr");
        return 1;
    }
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int inc = omp_get_num_threads();
        for (int i = id; i < ncb; i += inc) {
            double temp = 0;
            for (int j = 0; j < da; j++) {
                temp += a[j] * b[j][i];
            }
            output[i] = temp;
        }
    }
    return 0;
}

/**
 * couducts vector matrix left multiplication in parallel
 * @param  nra    number of rows of the matrix
 * @param  db     dimension of the vector/number of columns of the matrix
 * @param  a      the matrix
 * @param  b      the vector
 * @param  output the output vector
 * @return        whether the operation is successful
 */
int left_multiply(int nra, int db, double** a, double* b, double* output) {
    if (a == NULL || b == NULL || output == NULL) {
        puts("right multiply NULL ptr");
        return 1;
    }
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int inc = omp_get_num_threads();
        for (int i = id; i < nra; i += inc) {
            double temp = 0;
            for (int j = 0; j < db; j++) {
                temp += a[i][j] * b[j];
            }
            output[i] = temp;
        }
    }
    return 0;
}

/**
 * conducts vector addition on two vectors in parallel
 * @param  dim    dimension of the vectors
 * @param  a      left operand vector
 * @param  b      right operand vector
 * @param  output output vector
 * @return        whether the operation is successful
 */
int vector_add(int dim, double* a, double* b, double* output) {
    if (a == NULL || b == NULL || output == NULL) {
        puts("vector add NULL ptr");
        return 1;
    }
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int inc = omp_get_num_threads();
        for (int i = id; i < dim; i += inc) {
            output[i] = a[i] + b[i];
        }
    }
    return 0;
}

/**
 * conducts vector subtraction on two vectors in parallel
 * @param  dim    dimension of the vectors
 * @param  a      left operand vector
 * @param  b      right operand vector
 * @param  output output vector
 * @return        whether the operation is successful
 */
int vector_subtract(int dim, double* a, double* b, double* output) {
    if (a == NULL || b == NULL || output == NULL) {
        puts("vector subtract NULL ptr");
        return 1;
    }
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int inc = omp_get_num_threads();
        for (int i = id; i < dim; i += inc) {
            output[i] = a[i] - b[i];
        }
    }
    return 0;
}

/**
 * conducts matrix scalar multiplication in parallel
 * @param  nr     number of rows of the matrix
 * @param  nc     number of columns of the matrix
 * @param  a      the matrix
 * @param  factor scalar factor
 * @param  output output matrix
 * @return        whether the operation is successful
 */
int matrix_scalar_multiply(int nr, int nc, double** a, double factor, double** output) {
    if (a == NULL || output == NULL) {
        puts("matrix scalar multiply NULL ptr");
        return 1;
    }
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nc; j++) {
            output[i][j] = a[i][j] * factor;
        }
    }
    return 0;
}

/**
 * conducts vector scalar multiplication in parallel
 * @param  dim    dimension of the vector
 * @param  a      the vector
 * @param  factor scalar factor
 * @param  output output vector
 * @return        whether the operation is successful
 */
int vector_scalar_multiply(int dim, double* a, double factor, double* output) {
    if (a == NULL || output == NULL) {
        puts("vector scalar multiply NULL ptr");
        return 1;
    }
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int inc = omp_get_num_threads();
        for (int i = id; i < dim; i += inc) {
            output[i] = a[i] * factor;
        }
    }
    return 0;
}

/**
 * frees the memory allocated to a vector
 * @param  a   the vector
 * @return     whether the operation is successful
 */
int vector_free(double* a) {
    if (a == NULL) {
        puts("vector free NULL ptr");
        return 1;
    }
    free(a);
    return 0;
}

/**
 * frees the memory allocated to a matrix in parallel
 * @param  nr number of rows of the matrix
 * @param  a  the matrix
 * @return    whether the operation is successful
 */
int matrix_free(int nr, double** a) {
    if (a == NULL) {
        puts("free matrix NULL ptr");
        return 1;
    }
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int inc = omp_get_num_threads();
        for (int i = id; i < nr; i += inc) {
            free(a[i]);
        }
    }
    free(a);
    return 0;
}

/**
 * swaps column a in matrix a with column b in matrix b in parallel
 * @param  ca column number of the column in matrix a
 * @param  cb column nubmer of the column in matrix b
 * @param  nr number of rows of the matrices
 * @param  a  matrix a
 * @param  b  matrix b
 * @return    whether the operation is successful
 */
int swap_column(int ca, int cb, int nr, double** a, double** b) {
    if (a == NULL || b == NULL) {
        puts("swap column NULL ptr");
        return 1;
    }
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int inc = omp_get_num_threads();
        for (int i = id; i < nr; i += inc) {
            double temp = a[i][ca];
            a[i][ca] = b[i][cb];
            b[i][cb] = temp;
        }
    }
    return 0;
}

/**
 * swaps row a in matrix a with b in matrix b
 * @param  ra row number of the row in matrix a
 * @param  rb row number of the column in matrix b
 * @param  a  matrix a
 * @param  b  matrix b
 * @return    whether the operation is successful
 */
int swap_row(int ra, int rb, double** a, double** b) {
    if (a == NULL || b == NULL) {
        puts("swap row NULL ptr");
        return 1;
    }
    double* temp = a[ra];
    a[ra] = b[rb];
    b[rb] = temp;
    return 0;
}

/**
 * prints a representation of a vector
 * @param  dim    dimension of the vector
 * @param  vector the vector
 * @return        whether the operation is successful
 */
int print_vector(int dim, double* vector) {
    if (vector == NULL) {
        puts("print vector NULL ptr");
        return 1;
    }
    for (int i = 0; i < dim - 1; i++) {
        printf("%lf ", vector[i]);
    }
    printf("%lf\n", vector[dim - 1]);
    return 0;
}

/**
 * prints a representation of a matrix
 * @param  nr     number of rows of the matrix
 * @param  nc     number of columns of the matrix
 * @param  matrix the matrix
 * @return        whether the operation is successful
 */
int print_matrix(int nr, int nc, double** matrix) {
    if (matrix == NULL) {
        puts("print matrix NULL ptr");
        return 1;
    }
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nc - 1; j++) {
            printf("%lf ", matrix[i][j]);
        }
        printf("%lf\n", matrix[i][nc - 1]);
    }
    return 0;
}

/**
 * prints a representation of a vector with integer elements
 * @param  dim    dimension of the vector
 * @param  vector the vector
 * @return        whether the operation is successful
 */
int print_vector_int(int dim, int* vector) {
    if (vector == NULL) {
        puts("print vector NULL ptr");
        return 1;
    }
    for (int i = 0; i < dim - 1; i++) {
        printf("%d ", vector[i]);
    }
    printf("%d\n", vector[dim - 1]);
    return 0;
}

/**
 * prints a representation of a matrix with integer elements
 * @param  nr     number of rows of the matrix
 * @param  nc     number of columns of the matrix
 * @param  matrix the matrix
 * @return        whether the operation is successful
 */
int print_matrix_int(int nr, int nc, int** matrix) {
    if (matrix == NULL) {
        puts("print matrix NULL ptr");
        return 1;
    }
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nc - 1; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("%d\n", matrix[i][nc - 1]);
    }
    return 0;
}

/**
 * copies all elements in from matrix to to matrix in parallel
 * @param  nr   number of rows of the matrices
 * @param  nc   number of columns of the matrices
 * @param  from the from matrix
 * @param  to   the to matrix
 * @return      whether the operation is successful
 */
int matrix_copy(int nr, int nc, double** from, double** to) {
    if (from == NULL || to == NULL) {
        puts("matrix copy NULL ptr");
        return 1;
    }
    #pragma omp parallel for
    for (int i = 0; i < nr; i++) {
        memcpy(to[i], from[i], sizeof(double) * nc);
    }
    return 0;
}

/**
 * creates an identity matrix of dimension dim by dim in parallel
 * @param  dim dimension of the identity matrix to be created
 * @return     the created identity matrix
 */
double** create_identity(int dim) {
    double** opt = malloc(dim * sizeof(double*));
    if (opt == NULL) {
        puts("create identity malloc err");
        return NULL;
    }
    #pragma omp parallel for
    for (int i = 0; i < dim; i++) {
        opt[i] = calloc(dim, sizeof(double));
        if (opt[i] == NULL) {
            puts("create identity calloc err");
            exit(1);
        }
        opt[i][i] = 1;
    }
    return opt;
}

/**
 * conducts matrix inverse operation on square matrices using
 * Gaussian Elimination and back substitution in parallel
 * @param  dim    dimension of the matrix
 * @param  a      the matrix
 * @param  output output matrix
 * @return        whether the operation is successful
 */
int matrix_invert(int dim, double** a, double** output) {
    if (a == NULL || output == NULL) {
        puts("matrix invert NULL ptr");
        return 1;
    }
    //initialize the output matrix with elements in a matrix
    matrix_copy(dim, dim, a, output);
    //creates an identity matrix
    double** i_matrix = create_identity(dim);
    //gausian ellimination
    for (int i = 0; i < dim; i++) {
        if (output[i][i] == 0) {
            int j;
            for (j = i; j < dim && output[j][i] == 0; j++);
            //if no non zero element found, not invertible
            if (j == dim) {
                return -1;
            }
            swap_row(i, j, output, output);
        }
        double scale = output[i][i];
        #pragma omp parallel for
        for (int j = 0; j < dim; j++) {
            output[i][j] = output[i][j] / scale;
            i_matrix[i][j] = i_matrix[i][j] / scale;
        }
        #pragma omp parallel for
        for (int j = i + 1; j < dim; j++) {
            double factor = output[j][i];
            #pragma omp parallel for
            for (int k = 0; k < dim; k++) {
                output[j][k] = output[j][k] - factor * output[i][k];
                i_matrix[j][k] = i_matrix[j][k] - factor * i_matrix[i][k];
            }
        }
    }
    //back substitution
    for (int i = dim - 1; i > 0; i--) {
        #pragma parallel for
        for (int j = i - 1; j > -1; j--) {
            double factor = output[j][i];
            #pragma omp parallel for
            for (int k = 0; k < dim; k++) {
                output[j][k] = output[j][k] - factor * output[i][k];
                i_matrix[j][k] = i_matrix[j][k] - factor * i_matrix[i][k];
            }
        }
    }
    //copies the elements in the "identity matrix" to the output matrix
    matrix_copy(dim, dim, i_matrix, output);
    //frees memory allocated for the identity matrix to prevent memory leak
    matrix_free(dim, i_matrix);
    return 0;
}

/**
 * finds the maximum value and the associated index of a vector in parallel
 * @param  dim     dimension of the vector
 * @param  vector  the vector
 * @param  max_ind the index of the maximum value
 * @param  max     the maximum value
 * @return         whether the operation is successful
 */
int find_max(int dim, double* vector, int* max_ind, double* max) {
    if (vector == NULL || max == NULL) {
        puts("find max NULL ptr");
        return 1;
    }
    *max = vector[0];
    *max_ind = 0;
    #pragma omp parallel for
    for (int i = 1; i < dim; i++) {
        if (vector[i] >= *max) {
            *max_ind = i;
            *max = vector[i];
        }
    }
    return 0;
}

/**
 * extracts a column of a matrix and saves in a vector in parallel
 * @param  nr     number of rows of the matrix
 * @param  cn     column nubmer to extract
 * @param  matrix the matrix
 * @param  output the output vector
 * @return        whether the operation is successful
 */
int extract_column(int nr, int cn, double** matrix, double* output) {
    if (matrix == NULL || output == NULL) {
        puts("extract column NULL ptr");
        return 1;
    }
    #pragma omp parallel for
    for (int i = 0; i < nr; i++) {
        output[i] = matrix[i][cn];
    }
    return 0;
}

/**
 * computes the dot product of two vectors in parallel
 * @param  dim    the dimension of the vectors
 * @param  a      the left operand vector
 * @param  b      the right operand vector
 * @param  output the output vector
 * @return        whether the operation is successful
 */
int dot_product(int dim, double* a, double* b, double* output) {
    if (a == NULL || b == NULL || output == NULL) {
        puts("dot product NULL ptr");
        return 1;
    }
    *output = 0;
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int inc = omp_get_num_threads();
        double temp = 0;
        for (int i = id; i < dim; i += inc) {
            temp += a[i] * b[i];
        }
        //make sure to apply lock
        #pragma omp critical
        {
            *output += temp;
        }
    }
    return 0;
}

/**
 * finds the index of the first positive element in a vector
 * @param  dim    dimensionn of the vector
 * @param  vector the vector
 * @param  ind    the output index
 * @return        whether the operation is successful
 */
int find_first_pos(int dim, double* vector, int* ind) {
    *ind = -1;
    for (int i = 0; i < dim; i++) {
        if (vector[i] > 0) {
            *ind = i;
            return 0;
        }
    }
    return 0;
}

/**
 * finds the index of the first positive element in a vector after index start
 * @param  dim    dimension of the vector
 * @param  start  the start index
 * @param  vector the vector
 * @param  ind    the output index
 * @return        whether the operation is successful
 */
int find_next_pos(int dim, int start, double* vector, int* ind) {
    *ind = -1;
    for (int i = start + 1; i < dim; i++) {
        if (vector[i] > 0) {
            *ind = i;
            return 0;
        }
    }
    return 0;
}

/**
 * reallocates all rows in a matrix in parallel
 * @param  nr number of rows of the matrix
 * @param  nc number of columns of the matrix
 * @param  a  the matrix
 * @return    whether the operation is successful
 */
int matrix_realloc(int nr, int nc, double** a) {
    if (a == NULL) {
        puts("matrix realloc NULL ptr");
        fflush(stdout);
        return 1;
    }
    #pragma omp parallel for
    for (int i = 0; i < nr; i++) {
        a[i] = realloc(a[i], sizeof(double) * nc);
    }
    return 0;
}

/**
 * copies all elem of column a in matrix a to column b in matrix b
 * @param  nr  nuber of rows of the matrices
 * @param  acn column number in matrix a
 * @param  bcn column number in matrix b
 * @param  a   the a matrix
 * @param  b   the b matrix
 * @return     whether the operaiton is successful
 */
int copy_column(int nr, int acn, int bcn, double** a, double** b) {
    if (a == NULL || b == NULL) {
        puts("copy column NULL ptr");
        return 1;
    }
    #pragma omp parallel for
    for (int i = 0; i < nr; i++) {
        b[i][bcn] = a[i][acn];
    }
    return 0;
}
