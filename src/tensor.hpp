#ifndef TENSOR_H
#define TENSOR_H
#include <variant>
#include <vector>
class Tensor {
public:
  using NumericType = std::variant<int, float, double>;
  Tensor(NumericType val, int m, int n);
  Tensor(std::vector<NumericType> &data);
  Tensor(std::vector<std::vector<NumericType>> &data);
  Tensor(NumericType val, int n);
  Tensor(std::vector<NumericType> &data, int m, int n);
  int rows();
  int cols();
  std::vector<int> shape();
  Tensor operator+(NumericType scalar);
  Tensor operator*(NumericType scalar);
  Tensor operator-(NumericType scalar);
  Tensor operator/(NumericType scalar);
  Tensor operator%(NumericType scalar);

  Tensor operator+(Tensor other);
  Tensor operator*(Tensor other);
  Tensor operator-(Tensor other);
  Tensor operator/(Tensor other);
  Tensor operator%(Tensor other);

private:
  std::vector<int> view;
  std::vector<NumericType> data;
};

#endif
