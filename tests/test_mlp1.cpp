#include "compugraph.h"

class MLP1: public Model {
public:
	MLP1() {
		fc1 = new Dense(2, 2, LEAKY_RELU);
		fc2 = new Dense(2, 2);
	}
	Node *forward(Node *X) {
		X = fc1->forward(X);
		X = fc2->forward(X);
		auto Y = new Softmax(X);
		return Y;
	}
private:
	Model *fc1;
	Model *fc2;
};

void test_mlp1() {
	Graph::initInstance();

	// Create red points centered at (-2, -2)
	Tensor red_points = xt::random::randn<double>({50, 2}) - 2 * xt::ones<double>({50, 2});
	// Create blue points centered at (2, 2)
	Tensor blue_points = xt::random::randn<double>({50, 2}) + 2 * xt::ones<double>({50, 2});

	// Build input
	Tensor X = xt::concatenate(xtuple(blue_points, red_points));
	Tensor Y = xt::zeros<double>({100, 2});
	for (unsigned i = 0; i < 50; ++i) xt::index_view(Y, {{i, 0}}) = 1;
	for (unsigned i = 50; i < 100; ++i) xt::index_view(Y, {{i, 1}}) = 1;

	// MLP model
	MLP1 *model = new MLP1();
	model->compile(new GradientDescentOptimizer(0.03), new CrossEntropy(), { LOSS, ACCURACY });
	model->fit(X, Y, 5, 2, true, true);
	model->evaluate(X, Y, 1, true);
	Tensor x({2, 2});
	cout << "Prediction of (2, 2): " << model->predict(x) << endl;
	cout << "Prediction of (2, 2) in index: " << model->predict_index(x) << endl;
	
	Graph::deleteInstance();
}
