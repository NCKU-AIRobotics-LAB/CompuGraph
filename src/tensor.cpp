#include "tensor.h"

Dataset::Dataset(Tensor X, Tensor Y, int batch_size, bool shuffle): m_X(X), m_Y(Y), m_batch_size(batch_size), m_shuffle(shuffle) {
	m_index = 0;
	if (m_X.shape()[0] != m_Y.shape()[0]) throw("Size of X is different from size of Y!");
	if (m_batch_size < 0) {
		m_batch_size = 1;
		m_X = xt::view(m_X, xt::newaxis(), xt::all());
		m_Y = xt::view(m_Y, xt::newaxis(), xt::all());
	}
	if (m_X.shape()[0] < m_batch_size) throw("Size of data is smaller from batch size!");
	m_size = m_X.shape()[0];
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
	
	return Batch({ .X = xt::view(m_X, xt::keep(ids), xt::all()), .Y = xt::view(m_Y, xt::keep(ids), xt::all()), .size = m_batch_size, .epoch = m_epoch });
}
