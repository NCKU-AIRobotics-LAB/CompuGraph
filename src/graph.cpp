#include <stack>
#include "graph.h"

static Graph *m_instance = nullptr;

Graph::Graph() {}

Graph::~Graph() {
	for (auto node: m_placeholders) if (node != nullptr) delete node;
	for (auto node: m_variables) if (node != nullptr) delete node;
	for (auto node: m_operations) if (node != nullptr) delete node;
}

void Graph::addPlaceholder(Node *node) {
	m_placeholders.push_back(node);
}

void Graph::addVariable(Node *node) {
	m_variables.push_back(node);
}

void Graph::addOperation(Node *node) {
	m_operations.push_back(node);
}

static void recursive_nodes(Node *node, vector<Node *> &nodes_postorder) {
	if (node->getType() == OP) {
		for (auto input_node: dynamic_cast<Operation *>(node)->getInputNodes()) {
			recursive_nodes(input_node, nodes_postorder);
		}
	}
	nodes_postorder.push_back(node);
}

Tensor Graph::run(Node *operation, map<string, Tensor> feed_dict) {
	// We want to reach the leaves (end nodes) to get the value first => DFS
	// Get traversed post-order nodes (computational graph is a DAG)
	vector<Node *> nodes_postorder;
	recursive_nodes(operation, nodes_postorder);

	// Traverse all the nodes in post-order
	for (auto node: nodes_postorder) {
		if (node->getType() == PLACEHOLDER) {
			Placeholder *placeholder = dynamic_cast<Placeholder *>(node);
			node->setOutput(feed_dict[placeholder->getName()]);
		} else if (node->getType() == OP) {
			Operation *op = dynamic_cast<Operation *>(node);
			node->setOutput(op->compute());
		}
	}

	return operation->getOutput();
}

Graph *Graph::initInstance() {
	xt::random::seed(time(NULL));
	if (m_instance != nullptr) delete m_instance;
	m_instance = new Graph();
	return m_instance;
}

Graph *Graph::getInstance() {
	if (m_instance == nullptr) { // if not yet instantiated
		initInstance();	// create one and only object
	}
	return m_instance;
}

void Graph::deleteInstance() {
	if (m_instance != nullptr) delete m_instance;
	m_instance = nullptr;
}
