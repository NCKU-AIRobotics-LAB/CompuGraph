#ifndef MODEL_H
#define MODEL_H

#include <set>
#include "graph.h"

enum Metric { LOSS, ACCURACY, MSE };

// Note: negative means None for mumeric parameter values if we expect it be non-negative

class Model {
public:
	Model();
	virtual Node *forward(Node *) = 0;
	Node *operator()(Node *);
	virtual void compile(Optimizer *optimizer, LossFunction *loss, set<Metric> metrics);
	map<string, double> compute_batch_metrics(Batch batch, Node *y_pred, bool print_result = false);
	map<string, double> compute_metrics(Dataset data, bool print_result = true, bool c = false);
	// validation_split: -1 if no validation, 0 if using validation data, (0, 1] if using split data
	// batch_size: -1 if no batch dimension
	// validation_batch_size: -1 if not set
	virtual vector<map<string, double>> fit(Tensor x, Tensor y, int batch_size = 1, int epochs = 1, bool shuffle = true, double validation_split = -1, Tensor validation_data_x = Tensor(), Tensor validation_data_y = Tensor(), int validation_batch_size = -2, int validation_freq = 1, int initial_epoch = 0, bool print_result = true);
	virtual map<string, double> evaluate(Tensor x, Tensor y, int batch_size = 1, bool print_result = true);
	virtual Tensor predict(Tensor x, bool batch = false);
	virtual int predict_index(Tensor x);
	virtual xt::xarray<int> predict_index_batch(Tensor x);
	virtual void print_weight() {}
protected:
	Optimizer *m_optimizer;
	LossFunction *m_loss;
	set<Metric> m_metrics;
	Node *X;
	Node *Y_true;
	Node *Y_pred;
	Operation *loss_fn;
	Operation *minimization;
};

class Dense: public Model {
public:
	Dense(int input_dim, int output_size, OpType activation = IDENTITY);
	Node *forward(Node *X);
protected:
	int m_input_dim;
	int m_output_size;
	Variable *m_weights;
	Variable *m_bias;
	OpType m_activation;
};

#endif
