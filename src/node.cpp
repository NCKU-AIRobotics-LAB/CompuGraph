#include <sstream>
#include "graph.h"

static int node_current_id = 0;

// Utilities //////////////////////////////////////////////////////////////////
string node_type_to_string(NodeType node_type) {
	switch (node_type) {
		case OP: return "Operation"; break;
		case PLACEHOLDER: return "Placeholder"; break;
		case VAR: return "Variable"; break;
		default: return "None"; break;
	}
}

string op_type_to_string(OpType op_type) {
	switch (op_type) {
		case ADD: return "Add"; break;
		case IDENTITY: return "Identity"; break;
		case LEAKY_RELU: return "LeakyRelu"; break;
		case LOG: return "Log"; break;
		case MAT_MUL: return "MatMul"; break;
		case MUL: return "Mul"; break;
		case NEG: return "Neg"; break;
		case REDUCE_SUM: return "ReduceSum"; break;
		case RELU: return "Relu"; break;
		case SIGMOID: return "Sigmoid"; break;
		case SOFTMAX: return "Softmax"; break;
		case GRADIENT: return "Gradient"; break;
		default: return "None"; break;
	}
}

// Node //////////////////////////////////////////////////////////////////
Node::Node(NodeType type): m_type(type) {
	m_node_id = node_current_id++;
}

Node::~Node() {}

string Node::getId() {
	stringstream ss;
	string id;
	ss << m_node_id; ss >> id;
	return id;
}

NodeType Node::getType() {
	return m_type;
}

Tensor &Node::getOutput() {
	return m_output;
}

void Node::setOutput(Tensor output) {
	m_output = output;
}

vector<Node *> &Node::getConsumers() {
	return m_consumers;
}

void Node::addConsumer(Node *node) {
	m_consumers.push_back(node);
}

string Node::toString(bool only_shape) {
	string s, sd;
	s += node_type_to_string(getType()) + " Node " + getId() + "\n";
	s += "Consumers: [ "; for(auto &node: m_consumers) s += node->getId() + " "; s += "]\n";
	s += "Output shape: ( "; for(auto &d: m_output.shape()) { stringstream ss; ss << d; ss >> sd; s += sd + " "; } s+= ")\n";
	if (!only_shape) { stringstream ss; ss << m_output; s += "Output tensor:\n" + ss.str() + "\n"; }
	return s;
}

// Placeholder //////////////////////////////////////////////////////////////////
Placeholder::Placeholder(string name): Node(PLACEHOLDER), m_name(name) {
	Graph::getInstance()->addPlaceholder(this);
}

string Placeholder::getName() {
	return m_name;
}

string Placeholder::toString(bool only_shape) {
	string s = Node::toString(only_shape);
	s += "Name: " + m_name + "\n";
	return s;
}

// Variable //////////////////////////////////////////////////////////////////
Variable::Variable(): Node(VAR) {
	m_output = Tensor();
	Graph::getInstance()->addVariable(this);
}

Variable::Variable(Tensor initial_value): Node(VAR) {
	m_output = initial_value;
	Graph::getInstance()->addVariable(this);
}

Tensor &Variable::getValue() {
	return m_output;
}

void Variable::setValue(Tensor value) {
	m_output = value;
}

string Variable::toString(bool only_shape) {
	return Node::toString(only_shape);
}

// Operation //////////////////////////////////////////////////////////////////
Operation::Operation(OpType op_type): Node(OP), m_op_type(op_type) {
	Graph::getInstance()->addOperation(this);
}
Operation::Operation(OpType op_type, Node *input_node): Node(OP), m_op_type(op_type) {
	m_input_nodes.push_back(input_node);
	for (auto input_node: getInputNodes())
		input_node->addConsumer(this);
	Graph::getInstance()->addOperation(this);
}

Operation::Operation(OpType op_type, Node *input_node1, Node *input_node2): Node(OP), m_op_type(op_type) {
	m_input_nodes.push_back(input_node1);
	m_input_nodes.push_back(input_node2);
	for (auto input_node: getInputNodes())
		input_node->addConsumer(this);
	Graph::getInstance()->addOperation(this);
}

Operation::Operation(OpType op_type, vector<Node *> input_nodes): Node(OP), m_input_nodes(input_nodes), m_op_type(op_type) {
	for (auto input_node: getInputNodes())
		input_node->addConsumer(this);
	Graph::getInstance()->addOperation(this);
}

vector<Tensor> Operation::gradient(Tensor grad_back) {
	return vector<Tensor>();
}

vector<Node *> &Operation::getInputNodes() {
	return m_input_nodes;
}

Operation *Operation::createOp(OpType op_type, Node *input_node) {
	Operation *op;
	if (op_type == IDENTITY) {
		return new Identity(input_node);
	} else if (op_type == SIGMOID) {
		op = new Sigmoid(input_node);
	} else if (op_type == RELU) {
		op = new Relu(input_node);
	} else if (op_type == LEAKY_RELU) {
		op = new LeakyRelu(input_node);
	}
	return op;
}

string Operation::toString(bool only_shape) {
	string s;
	s += toStringBefore(only_shape);
	s += toStringAfter(only_shape);
	return s;
}

string Operation::toStringBefore(bool only_shape) {
	string s;
	s += "Operation Node " + getId() + "\n";
	s += "Operation Type: " + op_type_to_string(m_op_type) + "\n";
	return s;
}

string Operation::toStringAfter(bool only_shape) {
	string s, sd;
	s += "Consumers: [ "; for(auto &node: m_consumers) s += node->getId() + " "; s += "]\n";
	s += "Output shape: ( "; for(auto &d: m_output.shape()) { stringstream ss; ss << d; ss >> sd; s += sd + " "; } s+= ")\n";
	if (!only_shape) { stringstream ss; ss << m_output; s += "Output tensor:\n" + ss.str() + "\n"; }

	s += "Input nodes: [ "; for(auto &n: m_input_nodes) s += n->getId() + " "; s += "]\n";
	for (auto &node: m_input_nodes) {
		s += "\n" + node->toString(only_shape);
	}
	return s;
}


// Add //////////////////////////////////////////////////////////////////
Add::Add(Node *input_node1, Node *input_node2): Operation(ADD, input_node1, input_node2) {}

Tensor Add::compute() {
	Tensor &x = getInputNodes()[0]->getOutput();
	Tensor &y = getInputNodes()[1]->getOutput();
	return x + y;
}

vector<Tensor> Add::gradient(Tensor grad_back) {
	Tensor &a = getInputNodes()[0]->getOutput();
	Tensor &b = getInputNodes()[1]->getOutput();

	Tensor grad_wrt_a = grad_back;
	Tensor grad_wrt_b = grad_back;

	// The following becomes relevant if a and b are of different shapes
	while (grad_wrt_a.shape().size() > a.shape().size())
		grad_wrt_a = xt::sum(grad_wrt_a, 0);
	for (int i = 0; i < a.shape().size(); ++i)
		if (a.shape()[i] == 1)
			grad_wrt_a = xt::sum(grad_wrt_a, i, xt::keep_dims);
	
	while (grad_wrt_b.shape().size() > b.shape().size())
		grad_wrt_b = xt::sum(grad_wrt_b, 0);
	for (int i = 0; i < b.shape().size(); ++i)
		if (b.shape()[i] == 1)
			grad_wrt_b = xt::sum(grad_wrt_b, i, xt::keep_dims);
	
	return { grad_wrt_a, grad_wrt_b };
}


// Identity //////////////////////////////////////////////////////////////////
Identity::Identity(Node *input_node): Operation(IDENTITY, input_node) {}

Tensor Identity::compute() {
	Tensor &x = getInputNodes()[0]->getOutput();
	return x;
}

vector<Tensor> Identity::gradient(Tensor grad_back) {
  return { grad_back };
}


// LeakyRelu //////////////////////////////////////////////////////////////////
LeakyRelu::LeakyRelu(Node *input_node): Operation(LEAKY_RELU, input_node) {}

Tensor LeakyRelu::compute() {
	Tensor &x = getInputNodes()[0]->getOutput();
	return xt::where(x > 0, x, 0.01 * x);
}

vector<Tensor> LeakyRelu::gradient(Tensor grad_back) {
	Tensor &x = getInputNodes()[0]->getOutput();
  return { xt::where(x > 0, grad_back, 0.01 * grad_back) };
}


// Log //////////////////////////////////////////////////////////////////
Log::Log(Node *input_node): Operation(LOG, input_node) {}

Tensor Log::compute() {
	Tensor &x = getInputNodes()[0]->getOutput();
	return xt::log(x);
}

vector<Tensor> Log::gradient(Tensor grad_back) {
	Tensor &x = getInputNodes()[0]->getOutput();
  return { grad_back / x };
}


// MatMul //////////////////////////////////////////////////////////////////
MatMul::MatMul(Node *input_node1, Node *input_node2): Operation(MAT_MUL, input_node1, input_node2) {}

Tensor MatMul::compute() {
	Tensor &A = getInputNodes()[0]->getOutput();
	Tensor &B = getInputNodes()[1]->getOutput();
	return xt::linalg::dot(A, B);
}

vector<Tensor> MatMul::gradient(Tensor grad_back) {
	Tensor &A = getInputNodes()[0]->getOutput();
	Tensor &B = getInputNodes()[1]->getOutput();
	return { xt::linalg::dot(grad_back, xt::transpose(B)), xt::linalg::dot(xt::transpose(A), grad_back) };
}


// Mul //////////////////////////////////////////////////////////////////
Mul::Mul(Node *input_node1, Node *input_node2): Operation(MUL, input_node1, input_node2) {}

Tensor Mul::compute() {
	Tensor &x = getInputNodes()[0]->getOutput();
	Tensor &y = getInputNodes()[1]->getOutput();
	return x * y;
}

vector<Tensor> Mul::gradient(Tensor grad_back) {
	Tensor &A = getInputNodes()[0]->getOutput();
  Tensor &B = getInputNodes()[1]->getOutput();
 	return { grad_back * B, grad_back * A };
}


// Neg //////////////////////////////////////////////////////////////////
Neg::Neg(Node *input_node): Operation(NEG, input_node) {}

Tensor Neg::compute() {
	Tensor &x = getInputNodes()[0]->getOutput();
	return -x;
}

vector<Tensor> Neg::gradient(Tensor grad_back) {
	return { -grad_back };
}


// ReduceSum //////////////////////////////////////////////////////////////////
ReduceSum::ReduceSum(Node *input_node): Operation(REDUCE_SUM, input_node) {
	m_axis = {};
}

ReduceSum::ReduceSum(Node *input_node, int axis): Operation(REDUCE_SUM, input_node) {
	m_axis = {axis};
}

ReduceSum::ReduceSum(Node *input_node, vector<int> axis): Operation(REDUCE_SUM, input_node) {
	m_axis = axis;
}

Tensor ReduceSum::compute() {
	Tensor &A = getInputNodes()[0]->getOutput();
	if (m_axis.size() == 0)
		return xt::sum(A);
	return xt::sum(A, m_axis);
}

vector<Tensor> ReduceSum::gradient(Tensor grad_back) {
	Tensor &A = getInputNodes()[0]->getOutput();
	if (m_axis.size() == 0) {
		return { xt::ones<double>(A.shape()) * grad_back };
	}
	xt::xarray<int>::shape_type output_shape;
	xt::xarray<int>::shape_type tile_scaling;
	for (int i = 0; i < A.shape().size(); i++) {
		bool found = false;  
		for (auto &idx: m_axis) {
			if (idx == i) { found = true; break; }
		}
		if (found) {
			output_shape.push_back(1);
			tile_scaling.push_back(A.shape()[i]);
		} else {
			output_shape.push_back(A.shape()[i]);
			tile_scaling.push_back(1);
		}
	}
	grad_back.reshape(output_shape);
	return { xt::tile(grad_back, tile_scaling) };
}

string ReduceSum::toString(bool only_shape) {
	string s , sa;
	s = toStringBefore(only_shape);
	s += "Axis: { "; for(auto &ax: m_axis) { stringstream ss; ss << ax; ss >> sa; s += sa + " "; } s+= "}\n";
	s += toStringAfter(only_shape);
	return s;
}


// Relu //////////////////////////////////////////////////////////////////
Relu::Relu(Node *input_node): Operation(RELU, input_node) {}

Tensor Relu::compute() {
	Tensor &x = getInputNodes()[0]->getOutput();
	return xt::where(x > 0, x, 0);
}

vector<Tensor> Relu::gradient(Tensor grad_back) {
	Tensor &x = getInputNodes()[0]->getOutput();
  return { xt::where(x > 0, grad_back, 0) };
}


// Sigmoid //////////////////////////////////////////////////////////////////
Sigmoid::Sigmoid(Node *input_node): Operation(SIGMOID, input_node) {}

Tensor Sigmoid::compute() {
	Tensor &a = getInputNodes()[0]->getOutput();
	return 1 / (1 + xt::exp(-a));
}

vector<Tensor> Sigmoid::gradient(Tensor grad_back) {
	Tensor &sigmoid = getOutput();
  return { grad_back * sigmoid * (1 - sigmoid) };
}


// Softmax //////////////////////////////////////////////////////////////////
Softmax::Softmax(Node *input_node): Operation(SOFTMAX, input_node) {}

Tensor Softmax::compute() {
	Tensor &a = getInputNodes()[0]->getOutput();
	return xt::exp(a) / xt::view(xt::sum(xt::exp(a), 1), xt::all(), xt::newaxis());
}

vector<Tensor> Softmax::gradient(Tensor grad_back) {
	Tensor &softmax = getOutput();
	Tensor tmp = xt::sum(grad_back * softmax, 1);
	return { (grad_back - tmp.reshape({-1, 1})) * softmax };
}


// Gradient //////////////////////////////////////////////////////////////////
Gradient::Gradient(GradientDescentOptimizer *optimizer, Operation *loss): Operation(GRADIENT), m_optimizer(optimizer), m_loss(loss) {}

Tensor Gradient::compute() {
	shared_ptr<map<Node *, Tensor>> grads_and_vars = m_optimizer->compute_gradients(m_loss);
	m_optimizer->apply_gradients(grads_and_vars);
	return Tensor();
}

string Gradient::toString(bool only_shape) {
	string s;
	s += toStringBefore(only_shape);
	s += m_optimizer->toString();
	s += "Loss function: Node " + m_loss->getId() + "\n";
	s += m_loss->toString(only_shape);
	s += toStringAfter(only_shape);
	return s;
}
