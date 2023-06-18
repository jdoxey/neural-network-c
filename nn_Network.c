#include <stdlib.h>	// malloc, free, rand, RAND_MAX, srand
#include <string.h>	// strlen, strcpy, strtok
#include <stdarg.h>	// va_list, va_start, va_arg
#include <time.h>	// time
#include <math.h>	// exp
#include <stdio.h>	// printf

#include "nn_Network.h"

// 'private' functions
double nn_Network__sigmoid(double input);
double nn_Network__derivativeOfSigmoid(double sigmoid, double unused);
double nn_Network__cost(double desiredOutput, double networkOutput);
double nn_Network__derivativeOfCost(double computedOutput, double desiredOutput);

nn_Network *nn_Network_alloc(char *layout) {
	nn_Network *this = malloc(sizeof(nn_Network));
	this->layerActivations = NULL;

	this->numberOfLayers = 1;	// starts at 1 because there will be one more layer than there are commas
	for (int i = 0; layout[i] != '\0'; i++) {
		if (layout[i] == ',') {
			this->numberOfLayers++;
		}
	}

	// Allocate memory for weights.
	// Technically we could have one less because the first layer doesn't have weights,
	// but this way keeps the index numbers as you would expect, and doesn't take up much extra space.
	this->layerWeights = malloc(sizeof(nn_Matrix *) * this->numberOfLayers);

	// Make a copy of `layout` string because strtok doesn't work on string literals
	int layoutStringLength = strlen(layout);
	char *layoutCopy = malloc(sizeof(char) * (layoutStringLength + 1));
	strcpy(layoutCopy, layout);
	// Determine number of nodes in each layer from `layout` string
	const char comma[2] = ",";
	char *singleLayerSizeString = strtok(layoutCopy, comma);
	for (int l = 0; l < this->numberOfLayers; l++) {
		int thisLayerSize = atoi(singleLayerSizeString);
		if (l == 0) {
			this->numberOfInputs = thisLayerSize;
		}
		else {
			// Each weights matrix uses rows to indicate which node in previous layer the connection is coming from,
			// and columns for which node in this layer the connection goes to, i.e. this first index is where the
			this->layerWeights[l] = nn_Matrix_alloc(
				l == 1 ? this->numberOfInputs : nn_Network_numberOfNodesAtLayerIndex(this, l - 1),
				thisLayerSize);
		}
		singleLayerSizeString = strtok(NULL, comma);
	}
	free(layoutCopy);

	return this;
}

void nn_Network_free(nn_Network *this) {
	// Starts at 1 because we didn't allocate weights for the first layer
	for (int l = 1; l < this->numberOfLayers; l++) {
		nn_Matrix_free(this->layerWeights[l]);
	}
	// If this network was used for training, layerActivations will be non NULL
	if (this->layerActivations != NULL) {
		for (int l = 1; l < this->numberOfLayers; l++) {
			nn_Matrix_free(this->layerActivations[l]);
		}
		free(this->layerActivations);
	}
	free(this->layerWeights);
	free(this);
}

// TODO: clean up memory
nn_Matrix *nn_Network_inference(nn_Network *this, nn_Matrix *inputs) {
	// allocate memory for activations at each layer
	this->layerActivations = malloc(sizeof(nn_Matrix *) * this->numberOfLayers);
	this->layerActivations[0] = inputs;
	// Increment through each layer 'forwards', calculating the intermediate weightedSums,
	// and activations which are stored for back propagation.
	// (starts at 1 becuase there are no weights at the input layer)
	for (int l = 1; l < this->numberOfLayers; l++) {
		// Calculate the weighted sums (dot product) of previous layer activations and weights at this level,
		// then calculate the 'activation' value by applying the sigmoid function to the result.
		this->layerActivations[l] = nn_Matrix_allocWithDotProductThenFunctionApplied(
				this->layerActivations[l - 1], this->layerWeights[l], nn_Network__sigmoid);
	}
	return this->layerActivations[this->numberOfLayers - 1];
}

nn_Matrix *nn_Network_inferenceWithValues(nn_Network *this, ...) {
	va_list argp;
	va_start(argp, this);
	nn_Matrix *outputs = nn_Network_inferenceWithValuesArgp(this, argp);
	va_end(argp);
	return outputs;
}

nn_Matrix *nn_Network_inferenceWithValuesArgp(nn_Network *this, va_list argp) {
	nn_Matrix *inputs = nn_Matrix_alloc(1, this->numberOfInputs);
		for (int i = 0; i < this->numberOfInputs; i++) {
		inputs->data[i] = va_arg(argp, double);
	}
	return nn_Network_inference(this, inputs);
}

// inferenceForTraining keeps the outputs/activations from each layer.
nn_Matrix *nn_Network_inferenceForTraining(nn_Network *this, nn_Matrix *inputs) {
	// allocate memory for activations at each layer
	this->layerActivations = malloc(sizeof(nn_Matrix *) * this->numberOfLayers);
	this->layerActivations[0] = inputs;
	// Increment through each layer 'forwards', calculating the intermediate weightedSums,
	// and activations which are stored for back propagation.
	// (starts at 1 becuase there are no weights at the input layer)
	for (int l = 1; l < this->numberOfLayers; l++) {
		// Calculate the weighted sums (dot product) of previous layer activations and weights at this level,
		// then calculate the 'activation' value by applying the sigmoid function to the result.
		this->layerActivations[l] = nn_Matrix_allocWithDotProductThenFunctionApplied(
				this->layerActivations[l - 1], this->layerWeights[l], nn_Network__sigmoid);
	}
	return this->layerActivations[this->numberOfLayers - 1];
}

double nn_Network_train(nn_Network *this, nn_Matrix *trainingDataInputs, nn_Matrix *trainingDataOutputs, double trainingIncrement) {
	// First do a forward pass (inference)
	nn_Matrix *inferenceOutputs = nn_Network_inferenceForTraining(this, trainingDataInputs);

	// Calculate a single, overall average cost
	double averageCost = nn_Matrix_singleAverageAfterApplyingFunction(inferenceOutputs, trainingDataOutputs, nn_Network__cost);

	// Then do a backward pass, iterating backwards through the network calculating updates for each of the
	// weights based on direction and magnitude of gradient of each weight with respect to the final error/cost.
	// Updates are calculated during backwards pass, but not applied until after the backward pass is complete.

	// Allocate some space to store the updates while the backward pass is in progress.
	nn_Matrix **layerUpdates = malloc(sizeof(nn_Matrix *) * this->numberOfLayers);

	nn_Matrix *deltas = NULL;
	nn_Matrix *previousDeltas = NULL;
	for (int layer = this->numberOfLayers - 1; layer >= 1; layer--) {	// only goes down to index 1 because layer[0] has no weights
		// Compute the deltas for this layer
		if (layer == this->numberOfLayers - 1) {
			// for the output layer, deltas are the derivative of cost function times derivative of sigmoid output
			deltas = nn_Matrix_allocByMultiplyingAfterApplyingFunctions(inferenceOutputs, trainingDataOutputs,
					nn_Network__derivativeOfCost, nn_Network__derivativeOfSigmoid);
		}
		else {
			// deltas for other layers are calculated by taking each node in the current layer and summing the deltas from the
			// previous layer times the weight from this layer to previous layer times the derivative of the activations.
			nn_Matrix *thisLayerActivations = this->layerActivations[layer];
			deltas = nn_Matrix_alloc(thisLayerActivations->rows, thisLayerActivations->columns);
			for (int column = 0; column < thisLayerActivations->columns; column++) {
				for (int example = 0; example < thisLayerActivations->rows; example++) {
					double sum = 0.0;
					double activation = nn_Matrix_get(this->layerActivations[layer], example, column);
					double derivativeOfActivation = activation * (1 - activation);
					for (int previousDeltaColumn = 0; previousDeltaColumn < previousDeltas->columns; previousDeltaColumn++) {
						sum += nn_Matrix_get(previousDeltas, example, previousDeltaColumn) *
								nn_Matrix_get(this->layerWeights[layer + 1], column, previousDeltaColumn) *
								derivativeOfActivation;

					}
					nn_Matrix_set(deltas, example, column, sum);
				}
			}
		}
		// Calculate the derivative of cost with respect to each weight in this layer.
		// This is calculated as the average across all examples, of the delta for a node in this layer for a weight,
		// times the activation for the corresponding node from the previous layer corresponding to the same weight.
		layerUpdates[layer] = nn_Matrix_alloc(this->layerWeights[layer]->rows, this->layerWeights[layer]->columns);
		for (int weightRow = 0; weightRow < this->layerWeights[layer]->rows; weightRow++) {
			for (int weightColumn = 0; weightColumn < this->layerWeights[layer]->columns; weightColumn++) {
				double weightTotal = 0;
				for (int example = 0; example < deltas->rows; example++) {
					weightTotal += nn_Matrix_get(deltas, example, weightColumn) *
							nn_Matrix_get(this->layerActivations[layer - 1], example, weightRow);
				}
				nn_Matrix_set(layerUpdates[layer], weightRow, weightColumn, weightTotal / deltas->rows);
			}
		}

		// store deltas to use next time around this loop
		if (previousDeltas) {
			nn_Matrix_free(previousDeltas);
		}
		previousDeltas = deltas;
	}
	nn_Matrix_free(previousDeltas);

	// apply updates
	for (int layer = 1; layer < this->numberOfLayers; layer++) {
		nn_Matrix *layerWeights = this->layerWeights[layer];
		int numberOfWeightsInLayer = layerWeights->rows * layerWeights->columns;
		for (int weight = 0; weight < numberOfWeightsInLayer; weight++) {
			layerWeights->data[weight] += layerUpdates[layer]->data[weight] * trainingIncrement;
		}
		nn_Matrix_free(layerUpdates[layer]);
	}
	free(layerUpdates);

	return averageCost;
}

int nn_Network_numberOfNodesAtLayerIndex(nn_Network *this, int layerIndex) {
	if (layerIndex == 0) {
		return this->numberOfInputs;
	}
	else {
		return this->layerWeights[layerIndex]->columns;
	}
}

void nn_Network_randomiseWeightsBetweenMinAndMax(nn_Network *this, double min, double max) {
	// Seed the random number generator with the number of seconds since epoch.
	// N.B. time() function on linux gives a value in seconds, so running this twice in the same
	// second will generate the same set of values for each run.
	srand(time(NULL));
	// Starts at 1 becuase there are no weights at the input layer
	for (int l = 1; l < this->numberOfLayers; l++) {
		for (int j = 0; j < this->layerWeights[l]->rows; j++) {
			for (int k = 0; k < this->layerWeights[l]->columns; k++) {
				double randomNumber = ((rand() / (double)RAND_MAX) * (max - min)) + min;
				nn_Matrix_set(this->layerWeights[l], j, k, randomNumber);
			}
		}
	}
}

double nn_Network__sigmoid(double input) {
	return 1 / (1 + exp(0 - input));
}

double nn_Network__derivativeOfSigmoid(double sigmoid, double unused) {
	return sigmoid * (1 - sigmoid);
}

double nn_Network__cost(double desiredOutput, double networkOutput) {
	return pow(desiredOutput - networkOutput, 2);
}

double nn_Network__derivativeOfCost(double computedOutput, double desiredOutput) {
	return 2 * (desiredOutput - computedOutput);
}
