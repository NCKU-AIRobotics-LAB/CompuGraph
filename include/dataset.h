#ifndef DATASET_H
#define DATASET_H

#include "tensor.h"

struct Batch {
	Tensor X;
	Tensor Y;
	int size;
};

class Dataset {
public:
	Dataset();
	Dataset(Tensor X, Tensor Y, int batch_size = 1, bool shuffle = true);
	Batch getBatch();
	int getEpoch();
	void setEpoch(int epoch = 0);
	int getStepNum();
	static vector<Dataset> split(Tensor X, Tensor Y, double validation_split = 0.2, bool shuffle = true, int batch_size = 1, int split_batch_size = -1);
protected:
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
