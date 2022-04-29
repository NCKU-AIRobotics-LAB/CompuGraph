#include "model.h"

Model::Model() {}

Node *Model::operator()(Node *X) {
	return forward(X);
}

void Model::compile(Optimizer *optimizer, LossFunction *loss, set<Metric> metrics) {
	m_optimizer = optimizer;
	m_loss = loss;
	m_metrics = metrics;

	X = new Placeholder("X");
	Y_true = new Placeholder("Y");
	Y_pred = forward(X);
	loss_fn = m_loss->foward(Y_true, Y_pred);
	minimization = m_optimizer->minimize(loss_fn);
}

map<string, double> Model::compute_batch_metrics(Batch batch, Node *y_pred, bool print_result) {
	int correct = 0;
	if (m_metrics.count(ACCURACY) == 1) {
		auto pred = xt::argmax(y_pred->getOutput(), 1);
		for (int i = 0; i < batch.size; ++i) {
			correct += batch.Y(i, pred[i]) == 1;
		}
	}
	double acc = (batch.size != 0)? (double)correct / batch.size: 0;
	map<string, double> batch_metrics;
	batch_metrics["accuracy"] = acc;

	if (print_result) {
		if (m_metrics.count(ACCURACY) == 1) cout << " - accuracy: " << batch_metrics["accuracy"];
	}
	return batch_metrics;
}

map<string, double> Model::compute_metrics(Dataset data, bool print_result, bool is_val) {
	string prefix = is_val? "val_": "";

	data.setEpoch(0);
	int total = 0;
	double total_loss_value = 0;
	double total_acc = 0;
	while (true) {
		Batch batch = data.getBatch();
		if (data.getEpoch() >= 1) break;
		map<string, Tensor> feed_dict = {
			{ "X", batch.X },
			{ "Y", batch.Y }
		};
		auto loss_value = Graph::run(loss_fn, feed_dict)();
		auto batch_metrics = compute_batch_metrics(batch, Y_pred);
		if (m_metrics.count(LOSS) == 1) {
			total_loss_value += loss_value * batch.size;
		}
		if (m_metrics.count(ACCURACY) == 1) {
			total_acc += batch_metrics["accuracy"] * batch.size;
		}
		total += batch.size;
	}
	map<string, double> metrics;
	metrics[prefix + "loss"] = (total != 0)? total_loss_value / total: 0;
	metrics[prefix + "accuracy"] = (total != 0)? total_acc / total: 0;

	if (print_result) {
		if (m_metrics.count(LOSS) == 1) cout << " - " << prefix << "loss: " << metrics[prefix + "loss"];
		if (m_metrics.count(ACCURACY) == 1) cout << " - " << prefix << "accuracy: " << metrics[prefix + "accuracy"];
	}
	return metrics;
}

static void print_progress(int batch_iter, int step_num) {
	int width = int(log10(step_num) + 1);
	int progress = batch_iter / (step_num / 30);
	cout << flush << "\r" << setw(width) << batch_iter + 1 << "/" << step_num;
	int eq_num = 0;
	if (0 < progress) eq_num = progress - 1;
	if (progress > 30) eq_num = 30;
	int point_num = 30 - progress;
	if (progress > 30) point_num = 0;
	cout << " \[";
	for (int i = 0; i < eq_num; i++) cout << "=";
	if (0 < progress && progress <= 30) cout << ">";
	for (int i = 0; i < point_num; i++) cout << ".";
	cout << "]";
}

vector<map<string, double>> Model::fit(Tensor x, Tensor y, int batch_size, int epochs, bool shuffle, double validation_split, Tensor validation_data_x, Tensor validation_data_y, int validation_batch_size, int validation_freq, int initial_epoch, bool print_result) {
	// Preparing for training and validation data
	Dataset train_data, val_data;
	if (validation_split <= 0) {
		train_data = Dataset(x, y, batch_size, shuffle);
		if (validation_split == 0) {
			val_data = Dataset(validation_data_x, validation_data_y, validation_batch_size, shuffle);
		}
	} else {
		auto splited_data = Dataset::split(x, y, validation_split, shuffle, batch_size, validation_batch_size);
		train_data = splited_data[0];
		val_data = splited_data[1];
	}

	int current_epoch = initial_epoch;
	train_data.setEpoch(initial_epoch);
	int batch_iter = 0;
	vector<map<string, double>> history;

	while (true) {
		Batch batch = train_data.getBatch();

		// An epoch is finished
		if (train_data.getEpoch() > current_epoch) {
			print_progress(batch_iter, train_data.getStepNum());
			auto metrics = compute_metrics(train_data, print_result, false);
			if (validation_split >= 0) {
				auto val_metrics = compute_metrics(val_data, print_result, true);
				metrics.insert(val_metrics.begin(), val_metrics.end());
			}
			history.push_back(metrics);
			cout << endl;
			
			// Next epoch
			current_epoch = train_data.getEpoch();
			batch_iter = 0;
		}
		if (train_data.getEpoch() >= epochs) break;

		if (current_epoch < epochs && batch_iter == 0 && print_result) {
			cout << "Epoch " << current_epoch + 1 << "/" << epochs << endl;
		}

		// Forward
		map<string, Tensor> feed_dict = {
			{ "X", batch.X },
			{ "Y", batch.Y }
		};
		auto batch_loss = Graph::run(loss_fn, feed_dict)();
		if (print_result && batch_iter % (train_data.getStepNum() / 30) == 0) {
			print_progress(batch_iter, train_data.getStepNum());
			if (m_metrics.count(LOSS) == 1) cout << " - loss: " << batch_loss;
			auto batch_metrics = compute_batch_metrics(batch, Y_pred, print_result);
		}

		// Backward
		Graph::run(minimization, feed_dict);

		++batch_iter;
	}
	return history;
}

map<string, double> Model::evaluate(Tensor x, Tensor y, int batch_size, bool print_result) {
	cout << "Testing" << endl;
	Dataset data(x, y, batch_size, false);
	int batch_iter = 0;
	while (true) {
		Batch batch = data.getBatch();

		// An epoch is finished
		if (data.getEpoch() > 0) {
			print_progress(batch_iter, data.getStepNum());
			auto metrics = compute_metrics(data, print_result, false);
			cout << endl;
			return metrics;
		}

		// Forward
		map<string, Tensor> feed_dict = {
			{ "X", batch.X },
			{ "Y", batch.Y }
		};
		auto batch_loss = Graph::run(loss_fn, feed_dict)();
		if (print_result && batch_iter % (data.getStepNum() / 30) == 0) {
			print_progress(batch_iter, data.getStepNum());
			if (m_metrics.count(LOSS) == 1) cout << " - loss: " << batch_loss;
			auto batch_metrics = compute_batch_metrics(batch, Y_pred, print_result);
		}

		++batch_iter;
	}
}

Tensor Model::predict(Tensor x, bool batch) {
	Tensor batch_X = x;
	if (!batch) {
		auto x_shape = x.shape(); x_shape.insert(x_shape.begin(), 1);
		batch_X.reshape(x_shape);
	}
	map<string, Tensor> feed_dict = {
		{ "X", batch_X }
	};
	auto Y = Graph::run(Y_pred, feed_dict);
	if (batch) return Y;
	return xt::strided_view(Y, { 0, xt::ellipsis() });
}

int Model::predict_index(Tensor x) {
	auto pred = xt::argmax(predict(x, false));
	return pred(0);
}

xt::xarray<int> Model::predict_index_batch(Tensor x) {
	auto pred = xt::argmax(predict(x, true), 1);
	return pred;
}

Dense::Dense(int input_dim, int output_size, OpType activation): m_input_dim(input_dim), m_output_size(output_size), m_activation(activation) {
	m_weights = new Variable(0.01 * xt::random::randn<double>({m_input_dim, m_output_size}));
	m_bias = new Variable(xt::zeros<double>({1, m_output_size}));
}

Node *Dense::forward(Node *X) {
	Node *y = new Add(new MatMul(X, m_weights), m_bias);
	y = Operation::createOp(m_activation, y);
	return y;
}
