.PHONY: test
test:
	cc -o nn_MatrixTest nn_MatrixTest.c nn_Matrix.c -lm
	./nn_MatrixTest
	rm nn_MatrixTest
	cc -o nn_NetworkTest nn_NetworkTest.c nn_Network.c nn_Matrix.c -lm
	./nn_NetworkTest
	rm nn_NetworkTest

example:
	cc -o example example.c nn_Network.c nn_Matrix.c -lm
