#ifndef __MATRIX__H__
#define __MATRIX__H__
double* vector_alloc(int dim);//allocates memory for a vector
double** matrix_alloc(int nr, int nc);//allocates memory for a matrix in parallel
double** matrix_calloc(int nr, int nc);
int matrix_realloc(int nr, int nc, double** a);//reallocates rows of a matrix
//conducts matrix multiplication in parallel
int matrix_multiply(int nra, int nca, int ncb, double** a, double** b, double** output);
//conducts vector matrix right multipliication in parallel
int right_multiply(int da, int ncb, double* a, double** b, double* output);
//conducts vector matrix left multiplication in parallel
int left_multiply(int nra, int db, double** a, double* b, double* output);
//conducts addition of two matrices in parallel
int matrix_add(int nr, int nc, double** a, double** b, double** output);
//conducts subtraction of two matrices in parallel
int matrix_subtract(int nr, int nc, double** a, double** b, double** output);
//conducts addition of two vectors in parallel
int vector_add(int dim, double* a, double* b, double* output);
//conducts subtraction of two vectors in parallel
int vector_subtract(int dim, double* a, double* b, double* output);
//conducts matrix scalar multiplication in parallel
int matrix_scalar_multiply(int nr, int nc, double** a, double factor, double** output);
//conducts vector scalar multiplication in parallel
int vector_scalar_multiply(int dim, double* a, double factor, double* output);
//conducts matrix inverse in parallel
int matrix_invert(int dim, double** a, double** output);
//frees the memory allocated to a vector
int vector_free(double* a);
//frees the memory allocated to a matrix in parallel
int matrix_free(int nr, double** a);
//swaps column a in matrix a with column b in matrix b in parallel (a, b can be the same matrix)
int swap_column(int ca, int cb, int nr, double** a, double** b);
//swaps row a in matrix a with row b in matrix b in parallel (a, b can be the same matrix)
int swap_row(int ra, int rb, double** a, double** b);
//prints a representation of a vector to the console
int print_vector(int dim, double* vector);
//prints a representation of a matrix to the console
int print_matrix(int nr, int nc, double** matrix);
//prints a representation of a vector with integer values to the console
int print_vector_int(int dim, int* vector);
//prints a representation of a matrix with integer values to the console
int print_matrix_int(int nr, int nc, int** matrix);
//copies all elements in from matrix to to matrix in parallel
int matrix_copy(int nr, int nc, double** from, double** to);
//finds the maximum value and the associated index in a vector
int find_max(int dim, double* vector, int* max_ind, double* max);
//extracts a column in a matrix and saves in a vector in parallel
int extract_column(int nr, int cn, double** matrix, double* output);
//copies all elements in a column in from matrix to a column in to matrix
int copy_column(int nr, int acn, int bcn, double** from, double** to);
//creates an identity matrix of dimension dim by dim in parallel
double** create_identity(int dim);
//computes the dot product of two vectors
int dot_product(int dim, double* a, double* b, double* output);
//finds the associated index of the first positive element in a vector
int find_first_pos(int dim, double* vector, int* ind);
//finds the associated index the first positive element in a vector after start
int find_next_pos(int dim, int start, double* vector, int* ind);
#endif