#include <stdlib.h>	// malloc, free
#include <stdarg.h>	// va_list, va_start, va_arg
#include <stdio.h>	// printf

#include "nn_Matrix.h"

nn_Matrix *nn_Matrix_alloc(int rows, int columns) {
	nn_Matrix *this = malloc(sizeof(nn_Matrix));
	this->rows = rows;
	this->columns = columns;
	this->data = malloc(sizeof(double) * rows * columns);
	return this;
}

nn_Matrix *nn_Matrix_allocWithValues(int rows, int columns, ...) {
	va_list argp;
	va_start(argp, columns);
	nn_Matrix *this = nn_Matrix_allocWithValuesArgp(rows, columns, argp);
	va_end(argp);
	return this;
}

nn_Matrix *nn_Matrix_allocWithValuesArgp(int rows, int columns, va_list argp) {
	nn_Matrix *this = nn_Matrix_alloc(rows, columns);
	int numberOfValues = rows * columns;
	for (int i = 0; i < numberOfValues; i++) {
		this->data[i] = va_arg(argp, double);
	}
	return this;
}

nn_Matrix *nn_Matrix_allocWithDotProduct(nn_Matrix *inputA, nn_Matrix *inputB) {
	return nn_Matrix_allocWithDotProductThenFunctionApplied(inputA, inputB, NULL);
}

nn_Matrix *nn_Matrix_allocWithDotProductThenFunctionApplied(nn_Matrix *inputA, nn_Matrix *inputB, double (*functionToApply)(double)) {
	nn_Matrix *this = nn_Matrix_alloc(inputA->rows, inputB->columns);
	// iterate through each row of input A, each one will correspond to a row in the output matrix
	for (int inputARow = 0; inputARow < inputA->rows; inputARow++) {
		// calculate the weighted sum for each column at this row, each of these columns will correspond to a column in the output matrix
		for (int inputBColumn = 0; inputBColumn < inputB->columns; inputBColumn++) {
			// accumulate values of each column in input A multiplied by corresponding row of input B
			double total = 0.0;
			for (int inputAColumnInputBRow = 0; inputAColumnInputBRow < inputA->columns; inputAColumnInputBRow++) {
				total += nn_Matrix_get(inputA, inputARow, inputAColumnInputBRow) *
						nn_Matrix_get(inputB, inputAColumnInputBRow, inputBColumn);
			}
			if (functionToApply != NULL) {
				total = functionToApply(total);
			}
			nn_Matrix_set(this, inputARow, inputBColumn, total);
		}
	}
	return this;
}

nn_Matrix *nn_Matrix_allocByMultiplyingAfterApplyingFunctions(nn_Matrix *inputA, nn_Matrix *inputB,
		double (*functionToApplyA)(double, double), double (*functionToApplyB)(double, double)) {
	nn_Matrix *this = nn_Matrix_alloc(inputA->rows, inputA->columns);
	int totalSize = this->rows * this->columns;
	for (int i = 0; i < totalSize; i++) {
		this->data[i] = functionToApplyA(inputA->data[i], inputB->data[i]) *
				functionToApplyB(inputA->data[i], inputB->data[i]);
	}
	return this;
}

nn_Matrix *nn_Matrix_free(nn_Matrix *this) {
	free(this->data);
	free(this);
}

nn_Matrix *nn_Matrix_updateEachElementWithFunction(nn_Matrix *this, double (*updateFunction)(double)) {
	int totalSize = this->rows * this->columns;
	for (int i = 0; i < totalSize; i++) {
		this->data[i] = updateFunction(this->data[i]);
	}
}

double nn_Matrix_get(nn_Matrix *this, int row, int column) {
	return this->data[this->columns * row + column];
}

void nn_Matrix_set(nn_Matrix *this, int row, int column, double value) {
	this->data[this->columns * row + column] = value;
}

void nn_Matrix_fillWithValues(nn_Matrix *this, ...) {
	va_list argp;
	va_start(argp, this);
	nn_Matrix_fillWithValuesArgp(this, argp);
	va_end(argp);
}

void nn_Matrix_fillWithValuesArgp(nn_Matrix *this, va_list argp) {
	int numberOfValues = this->rows * this->columns;
	for (int i = 0; i < numberOfValues; i++) {
		this->data[i] = va_arg(argp, double);
	}
}

double nn_Matrix_singleAverageAfterApplyingFunction(nn_Matrix *this, nn_Matrix *other, double (*functionToApply)(double, double)) {
	double total = 0.0;
	int matrixSize = this->rows * this->columns;
	for (int i = 0; i < matrixSize; i++) {
		total += functionToApply(this->data[i], other->data[i]);
	}
	return total / matrixSize;
}

void nn_Matrix_print(nn_Matrix *this) {
	int totalSize = this->rows * this->columns;
	for (int i = 0; i < totalSize; i++) {
		if (i != 0 && i % this->columns == 0) {
			printf("\n");
		}
		printf("% 1.3lf ", this->data[i]);
	}
	printf("\n");
}
