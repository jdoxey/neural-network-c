#include <stdio.h>

#include "nn_Network.h"
#include "nn_Matrix.h"

int main() {
	nn_Network *network = nn_Network_alloc("2, 3, 1");
	nn_Network_randomiseWeightsBetweenMinAndMax(network, -3.0, 3.0);

	nn_Matrix *trainingDataInputs = nn_Matrix_allocWithValues(4, 2,
		0.0, 0.0,
		0.0, 1.0,
		1.0, 0.0,
		1.0, 1.0
	);
	nn_Matrix *trainingDataOutputs = nn_Matrix_allocWithValues(4, 1,
		1.0,
		0.0,
		0.0,
		1.0
	);

	double error;
	for (int epoc = 0; epoc < 100000; epoc++) {
		error = nn_Network_train(network, trainingDataInputs, trainingDataOutputs, 1);
		if (epoc % 10000 == 0) {
			printf("Epoc %i, error: %lf\n", epoc, error);
		}
	}
	printf("Final error: %lf\n", error);

	nn_Matrix *output;
	output = nn_Network_inferenceWithValues(network, 0.0, 0.0);
	printf("Output for values 0, 0 is %lf\n", nn_Matrix_get(output, 0, 0));
	output = nn_Network_inferenceWithValues(network, 0.0, 1.0);
	printf("Output for values 0, 1 is %lf\n", nn_Matrix_get(output, 0, 0));
	output = nn_Network_inferenceWithValues(network, 1.0, 0.0);
	printf("Output for values 1, 0 is %lf\n", nn_Matrix_get(output, 0, 0));
	output = nn_Network_inferenceWithValues(network, 1.0, 1.0);
	printf("Output for values 1, 1 is %lf\n", nn_Matrix_get(output, 0, 0));

	nn_Matrix_free(trainingDataInputs);
	nn_Matrix_free(trainingDataOutputs);
	nn_Network_free(network);

	return 0;
}
