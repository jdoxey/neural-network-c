#ifndef __NN_MATRIX_H__
#define __NN_MATRIX_H__


#include <stdarg.h>	// va_list

typedef struct {
	int rows;
	int columns;
	double *data;
} nn_Matrix;

nn_Matrix *nn_Matrix_alloc(int rows, int columns);
nn_Matrix *nn_Matrix_allocWithValues(int rows, int columns, ...);
nn_Matrix *nn_Matrix_allocWithValuesArgp(int rows, int columns, va_list argp);
nn_Matrix *nn_Matrix_allocWithDotProduct(nn_Matrix *inputA, nn_Matrix *inputB);
nn_Matrix *nn_Matrix_allocWithDotProductThenFunctionApplied(nn_Matrix *inputA, nn_Matrix *inputB, double (*functionToApply)(double));
nn_Matrix *nn_Matrix_allocByMultiplyingAfterApplyingFunctions(nn_Matrix *inputA, nn_Matrix *inputB,
		double (*functionToApplyA)(double, double), double (*functionToApplyB)(double, double));
nn_Matrix *nn_Matrix_free(nn_Matrix *this);
double nn_Matrix_get(nn_Matrix *this, int row, int column);
void nn_Matrix_set(nn_Matrix *this, int row, int column, double value);
void nn_Matrix_fillWithValues(nn_Matrix *this, ...);
void nn_Matrix_fillWithValuesArgp(nn_Matrix *this, va_list argp);
double nn_Matrix_singleAverageAfterApplyingFunction(nn_Matrix *this, nn_Matrix *other, double (*functionToApply)(double, double));
void nn_Matrix_print(nn_Matrix *this);


#endif
