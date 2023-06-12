#include <assert.h>
#include <stdio.h>

#include "nn_Matrix.h"

// Test function used in test for nn_Matrix_updateElementwiseWithFunction
double addOne(double input) {
	return input + 1.0;
}

// Test function used in test for nn_Matrix_singleAverageAfterAplyingFunction
double add(double a, double b) {
	return a + b;
}

int main() {
	// Test nn_Matrix_alloc, scenario: basic
	{
		nn_Matrix *matrix = nn_Matrix_alloc(3, 2);
		assert(matrix->rows == 3);
		assert(matrix->columns == 2);
		nn_Matrix_set(matrix, 2, 1, 0.4);
		assert(nn_Matrix_get(matrix, 2, 1) == 0.4);
		nn_Matrix_free(matrix);
	}

	// Test nn_Matrix_allocWithValues, scenario: basic
	{
		nn_Matrix *matrix = nn_Matrix_allocWithValues(2, 2,
			0.0, 1.0,
			2.0, 3.0
		);
		assert(nn_Matrix_get(matrix, 0, 0) == 0.0);
		assert(nn_Matrix_get(matrix, 0, 1) == 1.0);
		assert(nn_Matrix_get(matrix, 1, 0) == 2.0);
		assert(nn_Matrix_get(matrix, 1, 1) == 3.0);
		nn_Matrix_free(matrix);
	}

	// Test nn_Matrix_allocWithDotProduct, scenario: basic
	{
		// Test using some pre-computed values
		nn_Matrix *inputA = nn_Matrix_allocWithValues(4, 2,
			1.0, 1.0,
			0.0, 1.0,
			1.0, 0.0,
			1.0, 1.0
		);
		nn_Matrix *inputB = nn_Matrix_allocWithValues(2, 3,
			-2.0, 0.0, 2.0,
			-1.0, 1.0, -2.0
		);
		nn_Matrix *result = nn_Matrix_allocWithDotProduct(inputA, inputB);
		// First row
		assert(nn_Matrix_get(result, 0, 0) == -3.0);
		assert(nn_Matrix_get(result, 0, 1) == 1.0);
		assert(nn_Matrix_get(result, 0, 2) == 0.0);
		// Second row
		assert(nn_Matrix_get(result, 1, 0) == -1.0);
		assert(nn_Matrix_get(result, 1, 1) == 1.0);
		assert(nn_Matrix_get(result, 1, 2) == -2.0);
		// Third row
		assert(nn_Matrix_get(result, 2, 0) == -2.0);
		assert(nn_Matrix_get(result, 2, 1) == 0.0);
		assert(nn_Matrix_get(result, 2, 2) == 2.0);
		// Last row
		assert(nn_Matrix_get(result, 3, 0) == -3.0);
		assert(nn_Matrix_get(result, 3, 1) == 1.0);
		assert(nn_Matrix_get(result, 3, 2) == 0.0);
		nn_Matrix_free(inputA);
		nn_Matrix_free(inputB);
		nn_Matrix_free(result);
	}

	// Test nn_Matrix_allocWithDotProductThenFunctionApplied, scenario: basic
	{
		// Test using same pre-computed values as Test nn_Matrix_allocWithDotProduct,
		// with 1 added to each element.
		nn_Matrix *inputA = nn_Matrix_allocWithValues(4, 2,
			1.0, 1.0,
			0.0, 1.0,
			1.0, 0.0,
			1.0, 1.0
		);
		nn_Matrix *inputB = nn_Matrix_allocWithValues(2, 3,
			-2.0, 0.0, 2.0,
			-1.0, 1.0, -2.0
		);
		nn_Matrix *result = nn_Matrix_allocWithDotProductThenFunctionApplied(inputA, inputB, addOne);
		// First row
		assert(nn_Matrix_get(result, 0, 0) == -2.0);
		assert(nn_Matrix_get(result, 0, 1) == 2.0);
		assert(nn_Matrix_get(result, 0, 2) == 1.0);
		// Second row
		assert(nn_Matrix_get(result, 1, 0) == 0.0);
		assert(nn_Matrix_get(result, 1, 1) == 2.0);
		assert(nn_Matrix_get(result, 1, 2) == -1.0);
		// Third row
		assert(nn_Matrix_get(result, 2, 0) == -1.0);
		assert(nn_Matrix_get(result, 2, 1) == 1.0);
		assert(nn_Matrix_get(result, 2, 2) == 3.0);
		// Last row
		assert(nn_Matrix_get(result, 3, 0) == -2.0);
		assert(nn_Matrix_get(result, 3, 1) == 2.0);
		assert(nn_Matrix_get(result, 3, 2) == 1.0);
		nn_Matrix_free(inputA);
		nn_Matrix_free(inputB);
		nn_Matrix_free(result);
	}

	// Test nn_Matrix_allocByMultiplyingAfterApplyingFunctions, scenario: basic
	{
		nn_Matrix *a = nn_Matrix_allocWithValues(2, 2,
			1.0, 2.0,
			3.0, 4.0
		);
		nn_Matrix *b = nn_Matrix_allocWithValues(2, 2,
			5.0, 6.0,
			7.0, 8.0
		);
		nn_Matrix *matrix = nn_Matrix_allocByMultiplyingAfterApplyingFunctions(a, b, add, add);
		assert(nn_Matrix_get(matrix, 0, 0) == 36.0);
		assert(nn_Matrix_get(matrix, 0, 1) == 64.0);
		assert(nn_Matrix_get(matrix, 1, 0) == 100.0);
		assert(nn_Matrix_get(matrix, 1, 1) == 144.0);
	}

	// Test nn_Matrix_get, scenario: basic
	{
		nn_Matrix *matrix = nn_Matrix_allocWithValues(2, 3,
			0.0, 1.0, 2.0,
			3.0, 4.0, 5.0
		);
		assert(nn_Matrix_get(matrix, 0, 0) == 0.0);
		assert(nn_Matrix_get(matrix, 0, 1) == 1.0);
		assert(nn_Matrix_get(matrix, 0, 2) == 2.0);
		assert(nn_Matrix_get(matrix, 1, 0) == 3.0);
		assert(nn_Matrix_get(matrix, 1, 1) == 4.0);
		assert(nn_Matrix_get(matrix, 1, 2) == 5.0);
		nn_Matrix_free(matrix);
	}

	// Test nn_Matrix_set, scenario: basic
	{
		nn_Matrix *matrix = nn_Matrix_alloc(2, 3);
		nn_Matrix_set(matrix, 0, 0, 0.0);
		nn_Matrix_set(matrix, 0, 1, 1.0);
		nn_Matrix_set(matrix, 0, 2, 2.0);
		nn_Matrix_set(matrix, 1, 0, 3.0);
		nn_Matrix_set(matrix, 1, 1, 4.0);
		nn_Matrix_set(matrix, 1, 2, 5.0);
		assert(nn_Matrix_get(matrix, 0, 0) == 0.0);
		assert(nn_Matrix_get(matrix, 0, 1) == 1.0);
		assert(nn_Matrix_get(matrix, 0, 2) == 2.0);
		assert(nn_Matrix_get(matrix, 1, 0) == 3.0);
		assert(nn_Matrix_get(matrix, 1, 1) == 4.0);
		assert(nn_Matrix_get(matrix, 1, 2) == 5.0);
		nn_Matrix_free(matrix);
	}

	// Test nn_Matrix_fillWithValues, scenario: basic
	{
		nn_Matrix *matrix = nn_Matrix_alloc(2, 3);
		nn_Matrix_fillWithValues(matrix,
			0.0, 1.0, 2.0,
			3.0, 4.0, 5.0
		);
		assert(nn_Matrix_get(matrix, 0, 0) == 0.0);
		assert(nn_Matrix_get(matrix, 0, 1) == 1.0);
		assert(nn_Matrix_get(matrix, 0, 2) == 2.0);
		assert(nn_Matrix_get(matrix, 1, 0) == 3.0);
		assert(nn_Matrix_get(matrix, 1, 1) == 4.0);
		assert(nn_Matrix_get(matrix, 1, 2) == 5.0);
		nn_Matrix_free(matrix);
	}

	// Test nn_Matrix_singleAverageAfterApplyingFunction, scenario: basic
	{
		nn_Matrix *matrix = nn_Matrix_allocWithValues(2, 2,
			0.0, 1.0,
			2.0, 3.0
		);
		nn_Matrix *other = nn_Matrix_allocWithValues(2, 2,
			4.0, 5.0,
			6.0, 7.0
		);
		double average = nn_Matrix_singleAverageAfterApplyingFunction(matrix, other, add);
		assert(average > 6.999 && average < 7.001);
		nn_Matrix_free(matrix);
		nn_Matrix_free(other);
	}

	return 0;
}
