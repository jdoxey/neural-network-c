# Simple Neural Network written in C

![Build status](https://github.com/jdoxey/neural-network-c/actions/workflows/main.yml/badge.svg)

This is a very basic neural network implementation written with simple C.

I really just created this to learn more about the internals of neural networks.
I'm making it public in case anyone else finds it helpful or useful.


## Features

- Allows abitrary number of layers, and nodes in each layer (feed-forward only)
- Processes multiple training examples at a time
- Good unit test coverage


## Improvement Potential

- Include bias nodes
- Paralellisation across CPU and/or GPU to increase speed
- Load and save weight values
- Choice of activation function at different layers


## Usage

1. Create/allocate a new neural network with any number of layers, and any number of nodes in each layer, e.g.

	``` C
	nn_Network *network = nn_Network_alloc("2, 3, 1");
	```

1. Randomise weights within the network

	``` C
	nn_Network_randomiseWeightsBetweenMinAndMax(network, -3.0, 3.0);
	```

1. Set up training input and output data, e.g.

	``` C
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
	```

1. Complete a training pass (forward, then backward pass with weight updates),

	``` C
	double error = nn_Network_train(network, trainingDataInputs, trainingDataOutputs, 0.3);
	```

	Complete this step until `error` is at an acceptable level.

1. Clean up memory,

	``` C
	nn_Matrix_free(trainingDataInputs);
	nn_Matrix_free(trainingDataOutputs);
	nn_Network_free(network);
	```

6. Link in the C `math` library when building, e.g.

	``` sh
	cc -o example example.c nn_Network.c nn_Matrix.c -lm
	```
