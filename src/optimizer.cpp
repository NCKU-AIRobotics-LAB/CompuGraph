#include <set>
#include <queue>
#include <algorithm>
#include "optimizer.h"

Optimizer::Optimizer() {}

GradientDescentOptimizer::GradientDescentOptimizer(double learning_rate): m_learning_rate(learning_rate) {}

Operation *GradientDescentOptimizer::minimize(Operation *loss) {
	return new Gradient(this, loss);
}

shared_ptr<map<Node *, Tensor>> GradientDescentOptimizer::compute_gradients(Operation *loss) {
	// Back Propagation
	// We want to get the value near the starting node first => BFS
	// Contain the gradient of the loss w.r.t. the node's output
	shared_ptr<map<Node *, Tensor>> grads_and_vars = make_shared<map<Node *, Tensor>>();
	// The gradient of the loss with respect to the loss is just 1
	(*grads_and_vars)[loss] = xt::ones<double>(loss->getOutput().shape());

	// Perform a BFS, backwards from the loss
	set<Node *> visited;
	queue<Node *> q;
	visited.insert(loss);
	q.push(loss);
	while (!q.empty()) {
		Node *node = q.front();
		q.pop();
		// traverse every nodes
		if (node != loss) {
			// Compute the gradient of the loss with respect to this node's output
			(*grads_and_vars)[node] = 0;
			// Iterate all consumers
			for (auto consumer_node: node->getConsumers()) {
				Operation *consumer = dynamic_cast<Operation *>(consumer_node);
				// Pass the gradient of the loss w.r.t. consumer's output
				// Get the gradient of the loss with respect to all of consumer's inputs
				auto lossgrads_wrt_consumer_inputs = consumer->gradient((*grads_and_vars)[consumer]);
				auto &input_nodes = consumer->getInputNodes();
				for (int i = 0; i < input_nodes.size(); ++i) {
					if (input_nodes[i] == node) { // Retrieve the index of node in consumer's inputs
						// Get the gradient of the loss with respect to node, and add to total gradient
						(*grads_and_vars)[node] += lossgrads_wrt_consumer_inputs[i];
					}
				}
			}
		}
		// Append each input node to the queue
		if (node->getType() == OP) {
			for (auto input_node: dynamic_cast<Operation *>(node)->getInputNodes()) {
				if (visited.count(input_node) == 0) { // not yet visited
					visited.insert(input_node);
					q.push(input_node);
				}
			}
		}
	}

	return grads_and_vars;
}

void GradientDescentOptimizer::apply_gradients(shared_ptr<map<Node *, Tensor>> grads_and_vars) {
	for (auto &item: *grads_and_vars) {
		auto node = item.first;
		if (node->getType() == VAR) {
			// Retrieve gradient for this variable
			auto grad = item.second;
			Variable *var = dynamic_cast<Variable *>(node);
			// Take a step along the direction of the negative gradient
			var->setValue(var->getValue() - m_learning_rate * grad);
		}
	}
}

double GradientDescentOptimizer::getLearningRate() {
	return m_learning_rate;
}

void GradientDescentOptimizer::setLearningRate(double learning_rate) {
	m_learning_rate = learning_rate;
}

string GradientDescentOptimizer::toString() {
	stringstream ss;
	string s, sl;
	s += "Optimizer: Gradient Descent\n";
	ss << m_learning_rate; ss >> sl;
	s += "learning rate: " + sl + "\n";
	return s;
}

// Loss Functions
LossFunction::LossFunction() {}

Operation *LossFunction::operator()(Node *y_true, Node *y_pred) {
	return foward(y_true, y_pred);
}


CrossEntropy::CrossEntropy() {}

Operation *CrossEntropy::foward(Node *y_true, Node *y_pred) {
	return new Neg(new ReduceSum(new ReduceSum(new Mul(y_true, new Log(y_pred)), 1)));
}