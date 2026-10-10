#include "functional/tensor.h"
#include <algorithm>
#include <cassert>

namespace functional {
Tensor::Tensor(std::vector<std::size_t> shape)
    : shape_(std::move(shape)), strides_(shape_.size()), numel_(1),
      data_(nullptr) {

  std::size_t stride = 1;
  for (std::size_t d = shape_.size(); d-- > 0;) {
    strides_[d] = stride;
    stride *= shape_[d];
  }

  numel_ = stride;
  data_ = new float[numel_]();
}

Tensor::~Tensor() { delete[] data_; }

Tensor::Tensor(const Tensor &other)
    : shape_(other.shape_), strides_(other.strides_), numel_(other.numel_),
      data_(new float[other.numel_]) {
  std::copy(other.data_, other.data_ + numel_, data_);
}

Tensor &Tensor::operator=(const Tensor &other) {

  if (this == &other) {
    return *this;
  }

  float *fresh = new float[other.numel_];

  std::copy(other.data_, other.data_ + other.numel_, fresh);

  delete[] data_;
  data_ = fresh;
  shape_ = other.shape_;
  strides_ = other.strides_;
  numel_ = other.numel_;
  return *this;
}

Tensor::Tensor(Tensor &&other) noexcept
    : shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
      numel_(other.numel_), data_(other.data_) {
  other.data_ = nullptr;
  other.numel_ = 0;
  other.shape_.clear();
  other.strides_.clear();
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
  if (this != &other) {
    delete[] data_;
    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    numel_ = other.numel_;
    data_ = other.data_;
    other.data_ = nullptr;
    other.numel_ = 0;
    other.shape_.clear();
    other.strides_.clear();
  }

  return *this;
}

void Tensor::fill(float value) { std::fill(data_, data_ + numel_, value); }

Tensor Tensor::clone() const { return *this; }

} // namespace functional
