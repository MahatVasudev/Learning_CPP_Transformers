#include <core/tokenized_tensor.h>
#include <initializer_list>

namespace core {

TokenizedTensor::TokenizedTensor(vector<size_t> shape)
    : shape_(std::move(shape)), stride_(shape.size()), numel_(1) {

  std::size_t stride = 1;
  for (std::size_t d = shape.size(); d-- > 0;) {
    stride_[d] = stride;
    stride *= shape[d];
  }
  numel_ = stride;
}

TokenizedTensor::TokenizedTensor(const TokenizedTensor &other)
    : shape_(other.shape_), stride_(other.stride_), numel_(other.numel_),
      data_(new std::int32_t[other.numel_]) {

  std::copy(other.data_.get(), other.data_.get() + numel_, data_.get());
}

TokenizedTensor::TokenizedTensor(TokenizedTensor &&other) noexcept
    : shape_(std::move(other.shape_)), stride_(std::move(other.stride_)),
      numel_(std::move(other.numel_)), data_(std::move(other.data_)) {
  other.numel_ = 0;
  other.shape_.clear();
  other.stride_.clear();
}

TokenizedTensor &TokenizedTensor::operator=(const TokenizedTensor &other) {
  if (this != &other) {
    TokenizedTensor tmp(other);
    *this = std::move(tmp);
  }

  return *this;
}

TokenizedTensor &TokenizedTensor::operator=(TokenizedTensor &&other) noexcept {
  if (this != &other) {
    *this = std::move(other);
  }

  return *this;
}

TokenizedTensor TokenizedTensor::from_value(vector<size_t> shape,
                                            const vector<int32_t> values) {
  TokenizedTensor tt(std::move(shape));

  if (values.size() != tt.numel_) {
    // TODO: TN 002
  }

  std::copy(values.begin(), values.end(), tt.data());

  return tt;
}

TokenizedTensor
TokenizedTensor::from_value(vector<size_t> shape,
                            const std::initializer_list<int32_t> values) {
  TokenizedTensor tt(std::move(shape));

  if (values.size() != tt.numel_) {
    // TODO: TN 002
  }

  std::copy(values.begin(), values.end(), tt.data());

  return tt;
}

TokenizedTensor TokenizedTensor::from_pointer(vector<size_t> shape,
                                              const int32_t *values,
                                              const size_t n) {
  TokenizedTensor tt(std::move(shape));

  if (n != tt.numel_) {
    // TODO: TN 002
  }

  std::copy(values, values + n, tt.data());

  return tt;
}

Tensor TokenizedTensor::to_tensor() {
  Tensor t(shape_);

  std::copy(data_.get(), data_.get() + numel_, t.data());

  return t;
}

void TokenizedTensor::reshape(vector<size_t> shape) {

  vector<size_t> new_stride(shape.size());

  std::size_t stride = 1;
  for (std::size_t d = shape.size(); d-- > 0;) {
    new_stride[d] = stride;
    stride *= shape[d];
  }
  if (numel_ != stride) {
    // TODO: TN 002
  }
  numel_ = stride;
  stride_ = std::move(new_stride);
  shape_ = std::move(shape);
}

} // namespace core
