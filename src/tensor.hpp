#ifndef TENSOR_H
#define TENSOR_H
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>
class Tensor {
public:
  using NumericType = std::variant<int, float, double>;
  Tensor(NumericType val, size_t m, size_t n) : view{m, n} {
    data_flat = std::vector<NumericType>(n * m);
  };

  Tensor(std::vector<std::vector<NumericType>> &data) : data_flat() {
    if (data.empty() || data[0].empty()) {
      throw std::invalid_argument("input must be by m x n vector");
    }
    view = {static_cast<size_t>(data.size()),
            static_cast<size_t>(data[0].size())};
    for (const std::vector<NumericType> &row : data) {
      for (const NumericType &col : row) {
        data_flat.push_back(col);
      }
    }
  }

  Tensor(std::vector<std::vector<NumericType>> &&data) {
    if (data.empty() || data[0].empty()) {
      throw std::invalid_argument("input must be by m x n vector");
    }
    view = {static_cast<size_t>(data.size()),
            static_cast<size_t>(data[0].size())};
    for (std::vector<NumericType> &row : data) {
      for (NumericType &col : row) {
        data_flat.push_back(std::move(col));
      }
    }
  };

  Tensor(NumericType val, size_t m) : Tensor(val, m, 1){};
  Tensor(std::vector<NumericType> &data, size_t m, size_t n) : view{m, n} {
    if (data.empty()) {
      throw std::invalid_argument("input must be a m x 1 vector");
    }
    if (data.size() != (m * n)) {
      throw std::invalid_argument("m * n must equal data.size()");
    }
    data_flat = data;
  }
  Tensor(std::vector<NumericType> &&data, size_t m, size_t n)
      : data_flat(), view{m, n} {
    if (data.empty()) {
      throw std::invalid_argument("input must be a m x 1 vector");
    }
    if (data.size() != (m * n)) {
      throw std::invalid_argument("m * n must equal data.size()");
    }
    data_flat = std::move(data);
  };
  int rows() { return view.rows; };
  int cols() { return view.cols; };
  std::vector<int> shape();

  Tensor operator+(NumericType scalar) {
    std::vector<NumericType> data = data_flat;
    std::transform(data.begin(), data.end(), data.begin(),
                   [scalar](NumericType x) {
                     return std::visit(
                         [&](auto &&arg) -> NumericType {
                           using T = std::decay_t<decltype(arg)>;
                           return NumericType(arg + std::get<T>(scalar));
                         },
                         x);
                   });
    return Tensor(std::move(data), view.rows, view.cols);
  }

  Tensor operator*(NumericType scalar) {
    std::vector<NumericType> data = data_flat;
    std::transform(data.begin(), data.end(), data.begin(),
                   [scalar](NumericType x) {
                     return std::visit(
                         [&](auto &&arg) -> NumericType {
                           using T = std::decay_t<decltype(arg)>;
                           return NumericType(arg * std::get<T>(scalar));
                         },
                         x);
                   });
    return Tensor(std::move(data), view.rows, view.cols);
  };

  Tensor operator-(NumericType scalar) {
    std::vector<NumericType> data = data_flat;
    std::transform(data.begin(), data.end(), data.begin(),
                   [scalar](NumericType x) {
                     return std::visit(
                         [&](auto &&arg) -> NumericType {
                           using T = std::decay_t<decltype(arg)>;
                           return NumericType(arg - std::get<T>(scalar));
                         },
                         x);
                   });
    return Tensor(std::move(data), view.rows, view.cols);
  };

  Tensor operator/(NumericType scalar) {
    std::vector<NumericType> data = data_flat;
    std::transform(data.begin(), data.end(), data.begin(),
                   [scalar](NumericType x) {
                     return std::visit(
                         [&](auto &&arg) -> NumericType {
                           using T = std::decay_t<decltype(arg)>;
                           return NumericType(arg / std::get<T>(scalar));
                         },
                         x);
                   });
    return Tensor(std::move(data), view.rows, view.cols);
  };
  Tensor operator+(Tensor other);
  Tensor operator*(Tensor other);
  Tensor operator-(Tensor other);
  Tensor operator/(Tensor other);

private:
  struct View {
    size_t rows;
    size_t cols;
  } view;
  std::vector<NumericType> data_flat;
  bool requires_grad = false;
};

#endif
