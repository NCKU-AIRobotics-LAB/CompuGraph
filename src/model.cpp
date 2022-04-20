#include "model.h"

Model::Model() {}

void Model::compile(Optimizer *optimizer, LossFunction *loss, set<Metric> metrics) {
	m_optimizer = optimizer;
	m_loss = loss;
	m_metrics = metrics;

	X = new Placeholder("X");
	Y_true = new Placeholder("Y");
	Y_pred = (*this)(X);
	loss_fn = (*m_loss)(Y_true, Y_pred);
	minimization = m_optimizer->minimize(loss_fn);
}

vector<map<string, double>> Model::fit(Tensor &x, Tensor &y, int batch_size, int epochs, bool shuffle, bool print_result) {
	Dataset data(x, y, batch_size, shuffle);
	vector<map<string, double>> results;
	int current_epoch = 0;
	double loss_value = 0;
	int total = 0;
	int correct = 0;
	while (true) {
		Batch batch = data.getBatch();
		if (batch.epoch > current_epoch) {
			current_epoch = batch.epoch;
			double acc = (total != 0)? (double)correct / total: 0;
			map<string, double> result;
			result["loss"] = loss_value;
			result["accuracy"] = acc;
			results.push_back(result);
			if (print_result) {
				cout << "Epoch: " << current_epoch;
				if (m_metrics.count(LOSS) == 1) cout << ", Loss: " << loss_value;
				if (m_metrics.count(ACCURACY) == 1) cout << ", Accuracy: " << acc;
				cout << endl;
			}
			loss_value = 0;
			int total = 0;
			int correct = 0;
		}
		if (batch.epoch >= epochs) break;
		map<string, Tensor> feed_dict = {
			{ "X", batch.X },
			{ "Y", batch.Y }
		};
		loss_value += Graph::run(loss_fn, feed_dict)(0);
		if (m_metrics.count(ACCURACY) == 1) {
			auto pred = xt::argmax(Y_pred->getOutput(), 1);
			for (int i = 0; i < batch.size; ++i) {
				correct += batch.Y(i, pred[i]) == 1;
			}
			total += batch.size;
		}
		Graph::run(minimization, feed_dict);
	}
	return results;
}

map<string, double> Model::evaluate(Tensor &x, Tensor &y, int batch_size, bool print_result) {
	Dataset data(x, y, batch_size, false);
	double loss_value;
	int total = 0;
	int correct = 0;
	while (true) {
		Batch batch = data.getBatch();
		if (batch.epoch >= 1) break;
		map<string, Tensor> feed_dict = {
			{ "X", batch.X },
			{ "Y", batch.Y }
		};
		loss_value += Graph::run(loss_fn, feed_dict)(0);
		if (m_metrics.count(ACCURACY) == 1) {
			auto pred = xt::argmax(Y_pred->getOutput(), 1);
			for (int i = 0; i < batch.size; ++i) {
				correct += batch.Y(i, pred[i]) == 1;
			}
			total += batch.size;
		}
	}
	double acc = (total != 0)? (double)correct / total: 0;
	map<string, double> result;
	result["loss"] = loss_value;
	result["accuracy"] = acc;
	if (print_result) {
		cout << "Testing result";
		if (m_metrics.count(LOSS) == 1) cout << ", Loss: " << loss_value;
		if (m_metrics.count(ACCURACY) == 1) cout << ", Accuracy: " << acc;
		cout << endl;
	}
	return result;
}

Tensor Model::predict(Tensor &x, bool batch) {
	Tensor batch_X = x;
	if (!batch) batch_X = xt::view(batch_X, xt::newaxis(), xt::all());
	map<string, Tensor> feed_dict = {
		{ "X", batch_X }
	};
	auto Y = Graph::run(Y_pred, feed_dict);
	if (batch) return Y;
	return xt::view(Y, 0, xt::all());
}

int Model::predict_index(Tensor &x) {
	auto pred = xt::argmax(predict(x, false));
	return pred(0);
}

xt::xarray<int> Model::predict_index_batch(Tensor &x) {
	auto pred = xt::argmax(predict(x, true), 1);
	return pred;
}

Dense::Dense(int input_dim, int output_size, OpType activation): m_input_dim(input_dim), m_output_size(output_size), m_activation(activation) {
	m_weights = new Variable(0.01 * xt::random::randn<double>({m_input_dim, m_output_size}));
	m_bias = new Variable(xt::zeros<double>({1, m_output_size}));
}

Node *Dense::operator()(Node *X) {
	Node *y = new Add(new MatMul(X, m_weights), m_bias);
	y = Operation::createOp(m_activation, y);
	return y;
}