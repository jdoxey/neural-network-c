#ifndef __NN_NETWORK_H__
#define __NN_NETWORK_H__


#include <stdarg.h>	// va_list
#include <stdbool.h>	// bool, true, false

#include "nn_Matrix.h"

typedef struct {
	int numberOfLayers;
	int numberOfInputs;
	nn_Matrix **layerWeights;
	nn_Matrix **layerActivations;
} nn_Network;

#define NN_ERROR_WRITE_FOPEN_FAIL	1
#define NN_ERROR_WRITE_LOCK_FILE	2

nn_Network *nn_Network_alloc(char *layout);
nn_Network *nn_Network_allocFromFile(char *filename);
void nn_Network_free(nn_Network *this);

nn_Matrix *nn_Network_inference(nn_Network *this, nn_Matrix *inputs);
nn_Matrix *nn_Network_inferenceWithValues(nn_Network *this, ...);
nn_Matrix *nn_Network_inferenceWithValuesArgp(nn_Network *this, va_list argp);
nn_Matrix *nn_Network_inferenceForTraining(nn_Network *this, nn_Matrix *inputs);
double nn_Network_train(nn_Network *this, nn_Matrix *trainingDataInputs, nn_Matrix *trainingDataOutputs, double trainingIncrement);

int nn_Network_numberOfNodesAtLayerIndex(nn_Network *this, int layerIndex);
void nn_Network_randomiseWeightsBetweenMinAndMax(nn_Network *this, double min, double max);

int nn_Network_writeToFile(nn_Network *this, char *filename);


#endif
