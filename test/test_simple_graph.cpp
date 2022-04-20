#include "graph.h"

void test_simple_graph() {
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

	Tensor a = {{1,2,3,4}};
	a.reshape({4,1});
	cout << xt::exp(a) / xt::view(xt::sum(xt::exp(a), 1), xt::all(), xt::newaxis()) << endl;
	for (auto& el : a.shape()) {std::cout << el << ", "; } cout << endl;
	cout << a.shape().size() << endl;

	Tensor H = {{1,2},{3,4}};
	auto output_shape = xt::empty<int>({H.shape().size()});
	for (int i = 0; i < H.shape().size(); ++i) output_shape[i] = H.shape()[i];
	xt::xarray<int> original_shape = output_shape;
	for (auto &idx: {0}) output_shape(idx) = 1;
	auto tile_scaling = original_shape / output_shape;
	cout << tile_scaling << endl;

	H.reshape({-1, 1});
	cout << H << endl;
	
	Graph::deleteInstance();
}