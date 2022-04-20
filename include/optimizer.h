#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <memory>
#include <map>
#include "node.h"

class Optimizer {
public:
	Optimizer();
	virtual Operation *minimize(Operation *loss) = 0;
	virtual string toString() = 0;
};

class GradientDescentOptimizer: public Optimizer {
public:
	GradientDescentOptimizer(double learning_rate = 1e-3);
	Operation *minimize(Operation *loss);
	shared_ptr<map<Node *, Tensor>> compute_gradients(Operation *loss);
	void apply_gradients(shared_ptr<map<Node *, Tensor>> grads_and_vars);
	double getLearningRate();
	void setLearningRate(double learning_rate);
	string toString();
private:
	double m_learning_rate;
};


// Loss Functions
class LossFunction {
public:
	LossFunction();
	virtual Operation *operator()(Node *y_true, Node *y_pred) = 0;
};

class CrossEntropy: public LossFunction { // with softmax layer
public:
	CrossEntropy();
	Operation *operator()(Node *y_true, Node *y_pred);
};

#endif
