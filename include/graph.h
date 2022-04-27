#ifndef GRAPH_H
#define GRAPH_H

#include "optimizer.h"

class Graph {
public:
	Graph();
	~Graph();
	void addPlaceholder(Node *node);
	void addVariable(Node *node);
	void addOperation(Node *node);
	static Tensor run(Node *operation, map<string, Tensor> feed_dict = {});
	static Graph *initInstance(); // put this before any graphs or models created
	static Graph *getInstance();
	static void deleteInstance(); // put this where you want to surely delete the whole graphs
protected:
	vector<Node *> m_placeholders;
	vector<Node *> m_variables;
	vector<Node *> m_operations;
};

#endif
