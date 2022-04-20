#ifndef _GRAPH_H
#define _GRAPH_H

#include "optimizer.h"

class Graph {
public:
	Graph();
	~Graph();
	void addPlaceholder(Node *node);
	void addVariable(Node *node);
	void addOperation(Node *node);
	static Tensor run(Node *operation, map<string, Tensor> feed_dict = {});
	static Graph *initInstance(); // put this at the top of the main function
	static Graph *getInstance();
	static void deleteInstance(); // put this at the end of the main function
private:
	vector<Node *> m_placeholders;
	vector<Node *> m_variables;
	vector<Node *> m_operations;
};

#endif
