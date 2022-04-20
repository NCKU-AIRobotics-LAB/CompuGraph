#ifndef _NODE_H
#define _NODE_H

#include <vector>
#include <string>
#include "tensor.h"
#include "optimizer.h"

class GradientDescentOptimizer;

enum NodeType {
	NODE_NONE, OP, PLACEHOLDER, VAR
};

string node_type_to_string(NodeType node_type);

enum OpType {
	OP_NONE, ADD, IDENTITY, LEAKY_RELU, LOG, MAT_MUL, MUL, NEG, REDUCE_SUM, RELU, SIGMOID, SOFTMAX, GRADIENT
};

string op_type_to_string(OpType op_type);


class Node {
public:
	Node();
	virtual ~Node();
	string getId();
	NodeType getType();
	Tensor &getOutput();
	void setOutput(Tensor output);
	vector<Node *> &getConsumers();
	void addConsumer(Node *node);
	virtual string toString(bool only_shape = true);
protected:
	int m_node_id;
	NodeType m_type;
	Tensor m_output;
	vector<Node *> m_consumers;
};


class Placeholder: public Node {
public:
	Placeholder(string name);
	string getName();
	virtual string toString(bool only_shape = true);
protected:
	string m_name;
};


class Variable: public Node {
public:
	Variable();
	Variable(Tensor initial_value);
	Tensor &getValue();
	void setValue(Tensor value);
	virtual string toString(bool only_shape = true);
};


class Operation: public Node {
public:
	Operation();
	Operation(Node *input_node);
	Operation(Node *input_node1, Node *input_node2);
	Operation(vector<Node *> input_nodes);
	OpType getOpType();
	virtual Tensor compute() = 0;
	virtual vector<Tensor> gradient(Tensor grad_back);
	vector<Node *> &getInputNodes();
	static Operation *createOp(OpType op_type, Node * input_node);
	virtual string toString(bool only_shape = true);
protected:
	OpType m_op_type;
	vector<Node *> m_input_nodes;
	string toStringBefore(bool only_shape = true);
	string toStringAfter(bool only_shape = true);
};

class Add: public Operation {
public:
	Add(Node * input_node1, Node * input_node2);
	Tensor compute();
	vector<Tensor> gradient(Tensor grad_back);
};

class Identity: public Operation {
public:
	Identity(Node * input_node);
	Tensor compute();
	vector<Tensor> gradient(Tensor grad_back);
};

class LeakyRelu: public Operation {
public:
	LeakyRelu(Node * input_node);
	Tensor compute();
	vector<Tensor> gradient(Tensor grad_back);
};

class Log: public Operation {
public:
	Log(Node * input_node);
	Tensor compute();
	vector<Tensor> gradient(Tensor grad_back);
};

class MatMul: public Operation {
public:
	MatMul(Node * input_node1, Node * input_node2);
	Tensor compute();
	vector<Tensor> gradient(Tensor grad_back);
};

class Mul: public Operation {
public:
	Mul(Node * input_node1, Node * input_node2);
	Tensor compute();
	vector<Tensor> gradient(Tensor grad_back);
};

class Neg: public Operation {
public:
	Neg(Node * input_node);
	Tensor compute();
	vector<Tensor> gradient(Tensor grad_back);
};

class ReduceSum: public Operation {
public:
	ReduceSum(Node * input_node);
	ReduceSum(Node * input_node, int axis);
	ReduceSum(Node * input_node, vector<int> axis);
	Tensor compute();
	vector<Tensor> gradient(Tensor grad_back);
	string toString(bool only_shape = true);
private:
	vector<int> m_axis;
};

class Relu: public Operation {
public:
	Relu(Node * input_node);
	Tensor compute();
	vector<Tensor> gradient(Tensor grad_back);
};

class Sigmoid: public Operation {
public:
	Sigmoid(Node * input_node);
	Tensor compute();
	vector<Tensor> gradient(Tensor grad_back);
};

class Softmax: public Operation {
public:
	Softmax(Node * input_node);
	Tensor compute();
	vector<Tensor> gradient(Tensor grad_back);
};

class Gradient: public Operation {
public:
	Gradient(GradientDescentOptimizer *optimizer, Operation *loss);
	Tensor compute();
	string toString(bool only_shape = true);
private:
	GradientDescentOptimizer *m_optimizer;
	Operation *m_loss;
};

#endif
