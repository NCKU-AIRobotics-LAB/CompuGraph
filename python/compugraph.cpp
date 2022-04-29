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
	Node *forward(Node *X) {
		X = fc_in->forward(X);
		X = fc_out->forward(X);
		auto Y = new Softmax(X);
		return Y;
	}
protected:
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
	Node *forward(Node *X) {
		X = fc_in->forward(X);
		X = fc_1->forward(X);
		X = fc_out->forward(X);
		auto Y = new Softmax(X);
		return Y;
	}
protected:
	Model *fc_in;
	Model *fc_1;
	Model *fc_out;
};


void test() {
	py::print("CompuGraph Testing Version 0.0.5");
}

void dot() {
	py::print("Test BLAS and LAPACK");
	Tensor t1 = {{1, 2}, {3, 4}};
	Tensor t2 = {{5, 6}, {7, 8}};
	cout << xt::linalg::dot(t1, t2) << endl;
}


vector<map<string, double>> run(Model *model, Tensor X_train, Tensor Y_train, Tensor X_test, Tensor Y_test, int epochs) {
	// cout << "X_train shape: "; for(auto &i: X_train.shape()) cout << i << ", "; cout << endl;
	// cout << "Y_train shape: "; for(auto &i: Y_train.shape()) cout << i << ", "; cout << endl;
	// cout << "X_test shape: "; for(auto &i: X_test.shape()) cout << i << ", "; cout << endl;
	// cout << "Y_test shape: "; for(auto &i: Y_test.shape()) cout << i << ", "; cout << endl;

	// Model
	model->compile(new GradientDescentOptimizer(0.01), new CrossEntropy(), { LOSS, ACCURACY });

	auto results = model->fit(X_train, Y_train, 128, epochs, true, 0.2);

	auto testing_result = model->evaluate(X_test, Y_test, 32);
	map<string, double> result;
	result["test_loss"] = testing_result["loss"];
	result["test_accuracy"] = testing_result["accuracy"];
	results.push_back(result);

	return results;
}

vector<map<string, double>> mlp1(xt::pyarray<double>& X_train, xt::pyarray<double>& Y_train, xt::pyarray<double>& X_test, xt::pyarray<double>& Y_test, int epochs) {
	Graph::initInstance();
	Model *model = new MLP1();
	auto results = run(model, X_train, Y_train, X_test, Y_test, epochs);
	Graph::deleteInstance();
	return results;
}

vector<map<string, double>> mlp2(xt::pyarray<double>& X_train, xt::pyarray<double>& Y_train, xt::pyarray<double>& X_test, xt::pyarray<double>& Y_test, int epochs) {
	Graph::initInstance();
	Model *model = new MLP2();
	auto results = run(model, X_train, Y_train, X_test, Y_test, epochs);
	Graph::deleteInstance();
	return results;
}


PYBIND11_MODULE(compugraph, m) {
	xt::import_numpy();
	
	m.doc() = "CompuGraph - Simple Deep Learning Framework with Computational Graph"; // optional module docstring

	m.def("test", &test, pyout(), "print the test version");
	m.def("dot", &dot, pyout(), "test BLAS and LAPACK");
	m.def("mlp1", &mlp1, pyout(), "MLP 1");
	m.def("mlp2", &mlp2, pyout(), "MLP 2");
}
