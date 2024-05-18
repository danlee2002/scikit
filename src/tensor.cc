#include "tensor.hpp"
#include <algorithm>
Tensor::Tensor(NumericType val, int m, int n) {
  view.push_back(m);
  view.push_back(n);
  data.resize(m * n, val);
}

Tensor::Tensor(NumericType val, int n) : Tensor(val, n, 1) {}

Tensor::Tensor(std::vector<NumericType> &data) {
  view.push_back(data.size());
  view.push_back(1);
  for (NumericType elem : data) {
    Tensor::data.push_back(elem);
  }
}

Tensor::Tensor(std::vector<std::vector<NumericType>> &data) {
  view.push_back(data.size());
  view.push_back(data[0].size());
  for (std::vector<NumericType> rows : data) {
    for (NumericType col : rows) {
      Tensor::data.push_back(col);
    }
  }
}

std::vector<int> Tensor::shape() { return view; }

int Tensor::rows() { return view[0]; }

int Tensor::cols() { return view[1]; }

Tensor Tensor::operator+(NumericType scalar) {
  std::vector<NumericType> transformed = Tensor::data;

  return Tensor(transformed, Tensor::view[0], Tensor::view[1]);
}
