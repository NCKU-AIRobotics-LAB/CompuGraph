#include "tensor.h"

Dataset::Dataset() {}

Dataset::Dataset(Tensor X, Tensor Y, int batch_size, bool shuffle): m_X(X), m_Y(Y), m_batch_size(batch_size), m_shuffle(shuffle) {
	m_index = 0;
	if (m_X.shape()[0] != m_Y.shape()[0]) {
		cout << "Size of X = " << m_X.shape()[0] << " is different from size of Y = " << m_Y.shape()[0] << "!" << endl;
		if (m_X.shape()[0] < m_Y.shape()[0]) m_size = m_X.shape()[0];
		else m_size = m_Y.shape()[0];
		cout << "Set data set size to " << m_size << endl;
	} else m_size = m_X.shape()[0];
	if (m_batch_size < 0) {
		m_batch_size = 1;
		// add a new axis at front
		auto X_shape = X.shape(); X_shape.insert(X_shape.begin(), 1);
		auto Y_shape = Y.shape(); Y_shape.insert(Y_shape.begin(), 1);
		m_X.reshape(X_shape);
		m_Y.reshape(Y_shape);
	}
	if (m_size < m_batch_size) {
		cout << "Size of data " << m_size << " is smaller from batch size " << m_batch_size << " !" << endl;
		m_batch_size = m_size;
		cout << "Set batch size to " << m_batch_size << endl;
	}
	if (m_shuffle) perm = xt::random::permutation(m_size);
	else perm = xt::arange(m_size);
	m_epoch = 0;
}

Batch Dataset::getBatch() {
	if (m_size - m_index < m_batch_size) {
		if (m_shuffle) perm = xt::random::permutation(m_size);
		m_index = 0;
		++m_epoch;
	}
	xt::xarray<int> ids = xt::view(perm, xt::range(m_index, m_index + m_batch_size));
	m_index += m_batch_size;
	
	return Batch({ .X = xt::dynamic_view(m_X, { xt::keep(ids), xt::ellipsis() }), .Y = xt::dynamic_view(m_Y, { xt::keep(ids), xt::ellipsis() }), .size = m_batch_size });
}

int Dataset::getEpoch() {
	return m_epoch;
}

void Dataset::setEpoch(int epoch) {
	m_epoch = epoch;
}

int Dataset::getStepNum() {
	if (m_size % m_batch_size == 0) return m_size / m_batch_size;
	return m_size / m_batch_size + 1;
}

vector<Dataset> Dataset::split(Tensor X, Tensor Y, double validation_split, bool shuffle, int batch_size, int split_batch_size) {
	if (split_batch_size < 0) split_batch_size = batch_size;
	int size;
	if (X.shape()[0] != Y.shape()[0]) {
		cout << "Size of X = " << X.shape()[0] << " is different from size of Y = " << Y.shape()[0] << "!" << endl;
		if (X.shape()[0] < Y.shape()[0]) size = X.shape()[0];
		else size = Y.shape()[0];
		cout << "Set data set size to " << size << endl;
	} else size = X.shape()[0];
	if (batch_size < 0) {
		// add a new axis at front
		auto X_shape = X.shape(); X_shape.insert(X_shape.begin(), 1);
		auto Y_shape = Y.shape(); Y_shape.insert(Y_shape.begin(), 1);
		X.reshape(X_shape);
		Y.reshape(Y_shape);
	}

	xt::xarray<int> perm = xt::random::permutation(size);
	if (validation_split < 0) validation_split = 0;
	if (validation_split > 1) validation_split = 1;
	int split_size = size * validation_split;

	Tensor X_train, Y_train, X_split, Y_split;
	if (shuffle) {
		xt::xarray<int> ids = xt::view(perm, xt::range(0, size - split_size));
		X_train = xt::dynamic_view(X, { xt::keep(ids), xt::ellipsis() });
		Y_train = xt::dynamic_view(Y, { xt::keep(ids), xt::ellipsis() });
		ids = xt::view(perm, xt::range(size - split_size, size));
		X_split = xt::dynamic_view(X, { xt::keep(ids), xt::ellipsis() });
		Y_split = xt::dynamic_view(Y, { xt::keep(ids), xt::ellipsis() });
	} else {
		X_train = xt::strided_view(X, { xt::range(0, size - split_size), xt::ellipsis() });
		Y_train = xt::strided_view(Y, { xt::range(0, size - split_size), xt::ellipsis() });
		X_split = xt::strided_view(X, { xt::range(size - split_size, size), xt::ellipsis() });
		Y_split = xt::strided_view(Y, { xt::range(size - split_size, size), xt::ellipsis() });
	}
	return { Dataset(X_train, Y_train, batch_size, shuffle), Dataset(X_split, Y_split, split_batch_size, shuffle) };
}
