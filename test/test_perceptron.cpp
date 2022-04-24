#include "compugraph.h"

void test_perceptron() {
	Graph::initInstance();

	// Create red points centered at (-2, -2)
	Tensor red_points = xt::random::randn<double>({50, 2}) - 2 * xt::ones<double>({50, 2});
	// Create blue points centered at (2, 2)
	Tensor blue_points = xt::random::randn<double>({50, 2}) + 2 * xt::ones<double>({50, 2});

	// Initialize weights randomly
	auto W = new Variable(xt::random::randn<double>({2, 2}));
	auto b = new Variable(xt::random::randn<double>({2}));

	// Create placeholder
	auto X = new Placeholder("X");
	auto c = new Placeholder("c");

	// Build perceptron
	auto p = new Softmax(new Add(new MatMul(X, W), b));

	// Build cross-entropy loss
	auto J = new Neg(new ReduceSum(new ReduceSum(new Mul(c, new Log(p)), 1)));

	// Build minimization op
	auto minimization_op = (new GradientDescentOptimizer(0.01))->minimize(J);

	// Build placeholder input
	Tensor X_val = xt::concatenate(xtuple(blue_points, red_points));
	Tensor c_val = xt::zeros<double>({100, 2});
	for (unsigned i = 0; i < 50; ++i) xt::index_view(c_val, {{i, 0}}) = 1;
	for (unsigned i = 50; i < 100; ++i) xt::index_view(c_val, {{i, 1}}) = 1;
	map<string, Tensor> feed_dict = {
			{"X", X_val},
			{"c", c_val}
	};

	// Perform 100 gradient descent steps
	for (int step = 0; step < 100; ++step) {
		auto J_value = Graph::run(J, feed_dict);
		if (step % 10 == 0)
			cout << "Step:" << step << " Loss:" << J_value << endl;
		Graph::run(minimization_op, feed_dict);
	}

	// Print final result
	auto W_value = Graph::run(W);
	cout << "Weight matrix:\n" << W_value << endl;
	auto b_value = Graph::run(b);
	cout << "Bias:\n" << b_value << endl;
	
	Graph::deleteInstance();
}