#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>
#include "compugraph.h"
namespace py = pybind11;
typedef py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect> pyout;

class MLP1: public Model {
public:
	MLP1() {
		fc_in = new Dense(784, 128, LEAKY_RELU);
		fc_out = new Dense(128, 10);
	}
	Node *operator()(Node *X) {
		X = (*fc_in)(X);
		X = (*fc_out)(X);
		auto Y = new Softmax(X);
		return Y;
	}
private:
	Model *fc_in;
	Model *fc_out;
};

class MLP2: public Model {
public:
	MLP2() {
		fc_in = new Dense(784, 128, LEAKY_RELU);
		fc_1 = new Dense(128, 64, LEAKY_RELU);
		fc_out = new Dense(64, 10);
	}
	Node *operator()(Node *X) {
		X = (*fc_in)(X);
		X = (*fc_1)(X);
		X = (*fc_out)(X);
		auto Y = new Softmax(X);
		return Y;
	}
private:
	Model *fc_in;
	Model *fc_1;
	Model *fc_out;
};

// vector<map<string, double>> run(Model *model, xt::pyarray<double>& _X_train, xt::pyarray<double>& _Y_train, xt::pyarray<double>& _X_test, xt::pyarray<double>& _Y_test, int epochs) {
// 	Graph::initInstance();

// 	Tensor X_train = _X_train, Y_train = _Y_train, X_test = _X_test, Y_test = _Y_test;
// 	// cout << "X_train shape: "; for(auto &i: X_train.shape()) cout << i << ", "; cout << endl;
// 	// cout << "Y_train shape: "; for(auto &i: Y_train.shape()) cout << i << ", "; cout << endl;
// 	// cout << "X_test shape: "; for(auto &i: X_test.shape()) cout << i << ", "; cout << endl;
// 	// cout << "Y_test shape: "; for(auto &i: Y_test.shape()) cout << i << ", "; cout << endl;

// 	// Model
// 	model->compile(new GradientDescentOptimizer(0.01), new CrossEntropy(), { LOSS, ACCURACY });
// 	vector<map<string, double>> results;
// 	for (int i = 0; i < epochs; ++i) {
// 		map<string, double> result;
// 		auto training_result = model->fit(X_train, Y_train, 32, 1)[0];
// 		result["train_loss"] = training_result["loss"];
// 		result["train_accuracy"] = training_result["accuracy"];
// 		auto testing_result = model->evaluate(X_test, Y_test, 1);
// 		result["test_loss"] = testing_result["loss"];
// 		result["test_accuracy"] = testing_result["accuracy"];
// 		results.push_back(result);
// 	}

// 	Graph::deleteInstance();
// 	return results;
// }

// vector<map<string, double>> mlp1(xt::pyarray<double>& X_train, xt::pyarray<double>& Y_train, xt::pyarray<double>& X_test, xt::pyarray<double>& Y_test, int epochs) {
// 	auto model = new MLP1();
// 	return run(model, X_train, Y_train, X_test, Y_test, epochs);
// }

// vector<map<string, double>> mlp2(xt::pyarray<double>& X_train, xt::pyarray<double>& Y_train, xt::pyarray<double>& X_test, xt::pyarray<double>& Y_test, int epochs) {
// 	auto model = new MLP2();
// 	return run(model, X_train, Y_train, X_test, Y_test, epochs);
// }

vector<map<string, double>> mlp1(xt::pyarray<double>& _X_train, xt::pyarray<double>& _Y_train, xt::pyarray<double>& _X_test, xt::pyarray<double>& _Y_test, int epochs) {
	Graph::initInstance();

	Tensor X_train = _X_train, Y_train = _Y_train, X_test = _X_test, Y_test = _Y_test;

	// Model
	Model *model = new MLP1();
	model->compile(new GradientDescentOptimizer(0.01), new CrossEntropy(), { LOSS, ACCURACY });
	vector<map<string, double>> results;
	for (int i = 0; i < epochs; ++i) {
		map<string, double> result;
		auto training_result = model->fit(X_train, Y_train, 32, 1)[0];
		result["train_loss"] = training_result["loss"];
		result["train_accuracy"] = training_result["accuracy"];
		auto testing_result = model->evaluate(X_test, Y_test, 1);
		result["test_loss"] = testing_result["loss"];
		result["test_accuracy"] = testing_result["accuracy"];
		results.push_back(result);
	}

	Graph::deleteInstance();
	return results;
}

vector<map<string, double>> mlp2(xt::pyarray<double>& _X_train, xt::pyarray<double>& _Y_train, xt::pyarray<double>& _X_test, xt::pyarray<double>& _Y_test, int epochs) {
	Graph::initInstance();

	Tensor X_train = _X_train, Y_train = _Y_train, X_test = _X_test, Y_test = _Y_test;

	// Model
	Model *model = new MLP2();
	model->compile(new GradientDescentOptimizer(0.01), new CrossEntropy(), { LOSS, ACCURACY });
	vector<map<string, double>> results;
	for (int i = 0; i < epochs; ++i) {
		map<string, double> result;
		auto training_result = model->fit(X_train, Y_train, 32, 1)[0];
		result["train_loss"] = training_result["loss"];
		result["train_accuracy"] = training_result["accuracy"];
		auto testing_result = model->evaluate(X_test, Y_test, 1);
		result["test_loss"] = testing_result["loss"];
		result["test_accuracy"] = testing_result["accuracy"];
		results.push_back(result);
	}

	Graph::deleteInstance();
	return results;
}


PYBIND11_MODULE(compugraph, m) {
		xt::import_numpy();
		
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("mlp1", &mlp1, pyout(), "MLP 1");
		m.def("mlp2", &mlp2, pyout(), "MLP 2");
}