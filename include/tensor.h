#ifndef TENSOR_H
#define TENSOR_H

#include <ctime>
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xpad.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xio.hpp"
#include "xtensor-blas/xlinalg.hpp"
using namespace std;

using Tensor = xt::xarray<double>;

struct Batch {
	Tensor X;
	Tensor Y;
	int size;
	int epoch;
};

class Dataset {
public:
	Dataset(Tensor X, Tensor Y, int batch_size = 1, bool shuffle = true);
	Batch getBatch();
private:
	Tensor m_X;
	Tensor m_Y;
	int m_batch_size;
	int m_index;
	int m_size;
	bool m_shuffle;
	xt::xarray<int> perm;
	int m_epoch;
};

#endif
