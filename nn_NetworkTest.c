#include <assert.h>
#include <stdio.h>

#include "nn_Network.h"

int main() {
	// Test nn_Network_alloc, scenario: basic
	{
		nn_Network *network = nn_Network_alloc("2, 3, 1");
		assert(network->numberOfLayers == 3);
		assert(nn_Network_numberOfNodesAtLayerIndex(network, 0) == 2);
		assert(nn_Network_numberOfNodesAtLayerIndex(network, 1) == 3);
		assert(network->layerWeights[1]->rows == 2);	// nodes at previous (input) layer
		assert(network->layerWeights[1]->columns == 3);	// nodes at this (index 1) layer
		assert(nn_Network_numberOfNodesAtLayerIndex(network, 2) == 1);
		assert(network->layerWeights[2]->rows == 3);	// nodes at previous (index 1) layer
		assert(network->layerWeights[2]->columns == 1);	// nodes at this (output) layer
		nn_Network_free(network);
	}

	// Test nn_Network_randomiseWeightsBetweenMinAndMax, scenario: basic
	{
		nn_Network *network = nn_Network_alloc("2, 3, 1");
		nn_Network_randomiseWeightsBetweenMinAndMax(network, -3.0, 3.0);
		for (int l = 1; l < network->numberOfLayers; l++) {
			nn_Matrix *layerWeights = network->layerWeights[l];
			for (int j = 0; j < layerWeights->rows; j++) {
				for (int k = 0; k < layerWeights->columns; k++) {
					assert(nn_Matrix_get(layerWeights, j, k) >= -3.0);
					assert(nn_Matrix_get(layerWeights, j, k) <= 3.0);
					assert(nn_Matrix_get(layerWeights, j, k) != 0.0);
				}
			}
		}
		nn_Network_free(network);
	}

	// Test nn_Network_inferenceForTraining, scenario: basic
	{
		// use previously calculated values (see 2-3-1_example_spreadsheet.ods)
		nn_Matrix *inputs = nn_Matrix_allocWithValues(4, 2,
			0.0, 0.0,
			0.0, 1.0,
			1.0, 0.0,
			1.0, 1.0
		);

		nn_Network *network = nn_Network_alloc("2, 3, 1");
		nn_Matrix_fillWithValues(network->layerWeights[1],
			-2.0, 0.0, 2.0,
			-1.0, 1.0, -2.0
		);
		nn_Matrix_fillWithValues(network->layerWeights[2],
			-1.0,
			0.0,
			1.0
		);
		nn_Matrix *outputs = nn_Network_inferenceForTraining(network, inputs);
		// verify outputs
		assert(nn_Matrix_get(outputs, 0, 0) > 0.499 && nn_Matrix_get(outputs, 0, 0) < 0.501);
		assert(nn_Matrix_get(outputs, 1, 0) > 0.462 && nn_Matrix_get(outputs, 1, 0) < 0.463);
		assert(nn_Matrix_get(outputs, 2, 0) > 0.681 && nn_Matrix_get(outputs, 2, 0) < 0.682);
		assert(nn_Matrix_get(outputs, 3, 0) > 0.611 && nn_Matrix_get(outputs, 3, 0) < 0.612);
		// verify intermediate activations
		nn_Matrix *layer1Acitvations = network->layerActivations[1];
		assert(nn_Matrix_get(layer1Acitvations, 0, 0) > 0.499 && nn_Matrix_get(layer1Acitvations, 0, 0) < 0.501);
		assert(nn_Matrix_get(layer1Acitvations, 0, 1) > 0.499 && nn_Matrix_get(layer1Acitvations, 0, 1) < 0.501);
		assert(nn_Matrix_get(layer1Acitvations, 0, 2) > 0.499 && nn_Matrix_get(layer1Acitvations, 0, 2) < 0.501);
		assert(nn_Matrix_get(layer1Acitvations, 1, 0) > 0.268 && nn_Matrix_get(layer1Acitvations, 1, 0) < 0.269);
		assert(nn_Matrix_get(layer1Acitvations, 1, 1) > 0.731 && nn_Matrix_get(layer1Acitvations, 1, 1) < 0.732);
		assert(nn_Matrix_get(layer1Acitvations, 1, 2) > 0.119 && nn_Matrix_get(layer1Acitvations, 1, 2) < 0.120);
		assert(nn_Matrix_get(layer1Acitvations, 2, 0) > 0.119 && nn_Matrix_get(layer1Acitvations, 2, 0) < 0.120);
		assert(nn_Matrix_get(layer1Acitvations, 2, 1) > 0.499 && nn_Matrix_get(layer1Acitvations, 2, 1) < 0.501);
		assert(nn_Matrix_get(layer1Acitvations, 2, 2) > 0.880 && nn_Matrix_get(layer1Acitvations, 2, 2) < 0.881);
		assert(nn_Matrix_get(layer1Acitvations, 3, 0) > 0.047 && nn_Matrix_get(layer1Acitvations, 3, 0) < 0.048);
		assert(nn_Matrix_get(layer1Acitvations, 3, 1) > 0.731 && nn_Matrix_get(layer1Acitvations, 3, 1) < 0.732);
		assert(nn_Matrix_get(layer1Acitvations, 3, 2) > 0.499 && nn_Matrix_get(layer1Acitvations, 3, 2) < 0.501);

		nn_Matrix_free(inputs);
		nn_Network_free(network);
	}

	// Test nn_Network_inferenceForTraining, scenario: basic
	{
		// use previously calculated values (see 2-3-2_example_spreadsheet.ods)
		nn_Matrix *trainingInputs = nn_Matrix_allocWithValues(4, 2,
			0.0, 0.0,
			0.0, 1.0,
			1.0, 0.0,
			1.0, 1.0
		);
		nn_Matrix *trainingOutputs = nn_Matrix_allocWithValues(4, 2,
			0.0, 1.0,
			1.0, 0.0,
			1.0, 0.0,
			0.0, 1.0
		);

		nn_Network *network = nn_Network_alloc("2, 3, 2");
		nn_Matrix_fillWithValues(network->layerWeights[1],
			-2.0, 0.0, 2.0,
			-1.0, 1.0, -2.0
		);
		nn_Matrix_fillWithValues(network->layerWeights[2],
			-1.0, 2.0,
			0.0, -2.0,
			1.0, -1.0
		);
		double error = nn_Network_train(network, trainingInputs, trainingOutputs, 0.3);
		assert(error > 0.280 && error < 0.281);

		// verify new weights - layer 1
		assert(nn_Matrix_get(network->layerWeights[1], 0, 0) > -2.0 && nn_Matrix_get(network->layerWeights[1], 0, 0) < -1.999);
		assert(nn_Matrix_get(network->layerWeights[1], 0, 1) > -0.005 && nn_Matrix_get(network->layerWeights[1], 0, 1) < -0.004);
		assert(nn_Matrix_get(network->layerWeights[1], 0, 2) > 1.992 && nn_Matrix_get(network->layerWeights[1], 0, 2) < 1.993);
		assert(nn_Matrix_get(network->layerWeights[1], 1, 0) > -1.005 && nn_Matrix_get(network->layerWeights[1], 1, 0) < -1.004);
		assert(nn_Matrix_get(network->layerWeights[1], 1, 1) > 0.997 && nn_Matrix_get(network->layerWeights[1], 1, 1) < 0.998);
		assert(nn_Matrix_get(network->layerWeights[1], 1, 2) > -2.007 && nn_Matrix_get(network->layerWeights[1], 1, 2) < -2.006);
		// verify new weights - layer 2
		assert(nn_Matrix_get(network->layerWeights[2], 0, 0) > -1.004 && nn_Matrix_get(network->layerWeights[2], 0, 0) < -1.003);
		assert(nn_Matrix_get(network->layerWeights[2], 0, 1) > 2.009 && nn_Matrix_get(network->layerWeights[2], 0, 1) < 2.010);
		assert(nn_Matrix_get(network->layerWeights[2], 1, 0) > -0.006 && nn_Matrix_get(network->layerWeights[2], 1, 0) < -0.005);
		assert(nn_Matrix_get(network->layerWeights[2], 1, 1) > -1.986 && nn_Matrix_get(network->layerWeights[2], 1, 1) < -1.985);
		assert(nn_Matrix_get(network->layerWeights[2], 2, 0) > 0.991 && nn_Matrix_get(network->layerWeights[2], 2, 0) < 0.992);
		assert(nn_Matrix_get(network->layerWeights[2], 2, 1) > -0.986 && nn_Matrix_get(network->layerWeights[2], 2, 1) < -0.985);

		nn_Matrix_free(trainingInputs);
		nn_Matrix_free(trainingOutputs);
		nn_Network_free(network);
	}

	return 0;
}
