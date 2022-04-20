#include "graph.h"

void test_graph1() {
	Graph::initInstance();

	// Create variables
	auto A = new Variable(
		{
			{1, 0},
			{0, -1}
		}
	);

	auto b = new Variable(
		{1, 1}
	);

	// Create placeholder
	auto x = new Placeholder("x");

	// Create hidden node y
	auto y = new MatMul({A, x});

	// Create output node z
	auto z = new Add({y, b});

	auto output = Graph::run(z, { {"x", {1, 2} } });

	cout << output << endl;

	cout << z->toString(false) << endl;
	
	Graph::deleteInstance();
}