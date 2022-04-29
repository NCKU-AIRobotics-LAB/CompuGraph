#ifndef TENSOR_H
#define TENSOR_H

#include <ctime>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor/xdynamic_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor-blas/xlinalg.hpp>
using namespace std;

using Tensor = xt::xarray<double>;

#endif
