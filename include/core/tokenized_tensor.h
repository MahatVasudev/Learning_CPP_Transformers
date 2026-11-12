#ifndef TOKENIZED_TENSOR_CORE_H

#define TOKENIZED_TENSOR_CORE_H
#include <core/tensor.h>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>

using std::size_t;
using std::vector;

namespace core {
// only store data from computation purposes, and to be then shifted to Tensor;
class TokenizedTensor {
  TokenizedTensor(vector<size_t> shape);

  ~TokenizedTensor() = default;                             // Destructor
  TokenizedTensor(const TokenizedTensor &other);            // Copy
  TokenizedTensor(TokenizedTensor &&other) noexcept;        // Move
  TokenizedTensor &operator=(const TokenizedTensor &other); // Copy assignment
  TokenizedTensor &
  operator=(TokenizedTensor &&other) noexcept; // Move assignment

  std::int32_t *data() { return data_.get(); }

  const std::int32_t *data() const { return data_.get(); }

  static TokenizedTensor from_value(vector<size_t> shape,
                                    const vector<int32_t> values);

  static TokenizedTensor
  from_value(vector<size_t> shape, const std::initializer_list<int32_t> values);

  static TokenizedTensor from_pointer(vector<size_t> shape,
                                      const int32_t *values, const size_t n);

  Tensor to_tensor();

  // no infer shape
  void reshape(vector<size_t> shape);

private:
  vector<size_t> shape_;
  vector<size_t> stride_;
  size_t numel_;

  std::shared_ptr<std::int32_t[]> data_;
};

} // namespace core

#endif // !TOKENIZED_TENSOR_CORE_H
