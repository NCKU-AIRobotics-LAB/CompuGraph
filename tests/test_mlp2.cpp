#include "compugraph.h"

class MLP2: public Model {
public:
	MLP2() {
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

void test_mlp2() {
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
	MLP2 *model = new MLP2();
	model->compile(new GradientDescentOptimizer(0.03), new CrossEntropy(), { LOSS, ACCURACY });
	vector<map<string, double>> results;
	for (int i = 0; i < 10; ++i) {
		map<string, double> result;
		auto training_result = model->fit(X, Y, 5, 1)[0];
		result["train_loss"] = training_result["loss"];
		result["train_accuracy"] = training_result["accuracy"];
		auto testing_result = model->evaluate(X, Y, 1);
		result["test_loss"] = training_result["loss"];
		result["test_accuracy"] = training_result["accuracy"];
		results.push_back(result);
		cout << "Epoch " << i << ": train_loss = " << result["train_loss"] << ", train_acc = " << result["train_accuracy"] << ", test_loss = " << result["test_loss"] << ", test_acc = " << result["test_accuracy"] << endl;
	}
	Tensor x({2, 2});
	cout << "Prediction of (2, 2): " << model->predict(x) << endl;
	cout << "Prediction of (2, 2) in index: " << model->predict_index(x) << endl;
	
	Graph::deleteInstance();
}
