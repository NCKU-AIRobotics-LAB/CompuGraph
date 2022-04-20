#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>
#include "model.h"
namespace py = pybind11;
typedef py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect> pyout;

class MLP1: public Model {
public:
	MLP1() {
		fc1 = new Dense(2, 2, LEAKY_RELU);
		fc2 = new Dense(2, 2);
	}
	Node *operator()(Node *X) {
		X = (*fc1)(X);
		X = (*fc2)(X);
		auto Y = new Softmax(X);
		return Y;
	}
private:
	Model *fc1;
	Model *fc2;
};

vector<map<string, double>> simplednn(xt::pyarray<double>& data, int epochs) {
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
	vector<map<string, double>> results;
	for (int i = 0; i < epochs; ++i) {
		map<string, double> result;
		auto training_result = model->fit(X, Y, 5, 1)[0];
		result["train_loss"] = training_result["loss"];
		result["train_accuracy"] = training_result["accuracy"];
		auto testing_result = model->evaluate(X, Y, 1);
		result["test_loss"] = training_result["loss"];
		result["test_accuracy"] = training_result["accuracy"];
		results.push_back(result);
	}
	Tensor x({2, 2});
	cout << "Prediction of (2, 2): " << model->predict(x) << endl;
	cout << "Prediction of (2, 2) in index: " << model->predict_index(x) << endl;
	
	Graph::deleteInstance();
	return results;
}


PYBIND11_MODULE(pysimplednn, m) {
		xt::import_numpy();
		
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("simplednn", &simplednn, pyout(), "A function that adds two numbers");
}