#include "functional/tensor.h"
#include "errors/errors.h"
#include <algorithm>
#include <cassert>
#include <string>
#include <utility>

namespace functional {
Tensor::Tensor(std::vector<std::size_t> shape)
    : shape_(std::move(shape)), strides_(shape_.size()), numel_(1),
      data_(nullptr) {

  strides_ = compute_strides_(shape_, numel_);
  data_ = new float[numel_]();
}

Tensor::~Tensor() { delete[] data_; }

Tensor::Tensor(const Tensor &other)
    : shape_(other.shape_), strides_(other.strides_), numel_(other.numel_),
      data_(new float[other.numel_]) {
  std::copy(other.data_, other.data_ + numel_, data_);
}

void Tensor::reshape(std::vector<std::size_t> new_shape) {
  std::size_t new_numel = 1;
  std::vector<std::size_t> new_strides = compute_strides_(new_shape, new_numel);

  if (new_numel != numel_) {
    throw error::ShapeError(
        "Tensor::reshape",
        error::make_message("a shape with ", numel_, " elements"), new_shape,
        error::make_message("Element Count mismatch: requested ", new_numel,
                            " elements, but Tensor has ", numel_, "elements"));
  }
  shape_ = std::move(new_shape);
  strides_ = std::move(new_strides);
}

template <typename T> bool bound_check(T &value, T lower_bound, T upper_bound) {
  return lower_bound <= value && value <= upper_bound;
}

int normalize_dim(int dim, std::size_t rank, bool allow_end = true) {
  const int r = static_cast<int>(rank);
  const int upper = allow_end ? r : r - 1;
  const int lower = allow_end ? -(r + 1) : -r;

  if (!bound_check(dim, lower, upper)) {
    throw error::IndexError(
        error::make_message("dim must be in between ", lower, " and ", upper),
        dim);
  }

  if (dim < 0) {
    dim += r + (allow_end ? 1 : 0);
  }

  return dim;
}

void Tensor::squeeze(int &dim) {
  std::size_t _rank = this->rank();
  const int dim_norm = normalize_dim(dim, _rank, false);

  if (shape_[dim] != 1) {
    throw error::ShapeError(
        "Tensor::squeeze",
        error::make_message("dimension ", dim_norm, "to have size 1"), shape_,
        error::make_message("Found dimension size ", shape_[dim_norm], '.'));
  }

  std::vector<std::size_t> new_shape = shape_;

  new_shape.erase(new_shape.begin() + dim_norm);

  this->reshape(std::move(new_shape));
}

void Tensor::unsqueeze(int &dim) {
  std::size_t _rank = this->rank();
  const int dim_norm = normalize_dim(dim, _rank, true);
  std::vector<std::size_t> new_shape = shape_;

  new_shape.insert(new_shape.begin() + dim_norm, 1);

  this->reshape(std::move(new_shape));
}

Tensor Tensor::squeeze_copy(int &dim) const {
  const int d = normalize_dim(dim, rank());

  if (shape_[d] != 1) {
    throw error::ShapeError(
        "Tensor::squeeze",
        error::make_message("dimension ", d, "to have size 1"), shape_,
        error::make_message("Found dimension size ", shape_[d], '.'));
  }

  std::vector<std::size_t> new_shape = shape_;

  new_shape.erase(new_shape.begin() + d);

  Tensor new_tensor(new_shape);

  std::copy(data_, data_ + numel_, new_tensor.data_);

  return new_tensor;
}

Tensor Tensor::unsqueeze_copy(int &dim) const {
  const int d = normalize_dim(dim, rank());
  std::vector<std::size_t> new_shape = shape_;

  new_shape.insert(new_shape.begin() + d, 1);
  Tensor new_tensor(new_shape);

  std::copy(data_, data_ + numel_, new_tensor.data_);

  return new_tensor;
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

Tensor Tensor::from_values(std::vector<std::size_t> shape,
                           const std::vector<float> &values) {
  Tensor t(std::move(shape));

  if (values.size() != t.numel())
    throw error::ShapeError(
        "Tensor::from_values",
        error::make_message("miss-match between the received shape and the "
                            "number of values, got values size: ",
                            values.size()),
        shape,
        error::make_message(
            "The expected number of values from the tensor are: ", t.numel()));

  std::copy(values.begin(), values.end(), t.data());

  return t;
}

Tensor Tensor::from_values(std::vector<std::size_t> shape,
                           std::initializer_list<float> values) {

  Tensor t(std::move(shape));

  if (values.size() != t.numel())
    throw error::ShapeError(
        "Tensor::from_values",
        error::make_message("miss-match between the received shape and the "
                            "number of values, got values size: ",
                            values.size()),
        shape,
        error::make_message(
            "The expected number of values from the tensor are: ", t.numel()));

  std::copy(values.begin(), values.end(), t.data());

  return t;
}

Tensor Tensor::from_pointer(std::vector<std::size_t> shape, const float *src,
                            const std::size_t count) {

  Tensor t(std::move(shape));
  assert(src != nullptr);
  assert(count == t.numel());
  std::copy(src, src + count, t.data());
  return t;
}

void Tensor::fill(float value) { std::fill(data_, data_ + numel_, value); }

Tensor Tensor::clone() const { return *this; }

std::vector<std::size_t>
Tensor::compute_strides_(std::vector<std::size_t> &shape, std::size_t &numel) {
  std::vector<std::size_t> strides(shape.size());
  std::size_t stride = 1;

  for (std::size_t d = shape.size(); d-- > 0;) {
    strides[d] = stride;
    stride *= shape[d];
  }

  numel = stride;
  return strides;
}

Tensor reshape(Tensor &tensor, std::vector<std::size_t> &shape) {
  Tensor t_diff = tensor;

  t_diff.reshape(shape);

  return t_diff;
}

} // namespace functional
