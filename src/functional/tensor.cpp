#include "functional/tensor.h"
#include "errors/errors.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void gather_logical(const float *src, const std::vector<std::size_t> &shape,
                    const std::vector<std::size_t> &strides, float *dst,
                    std::size_t numel) {
  const std::size_t rank = shape.size();
  std::vector<std::size_t> idx(rank, 0);

  std::size_t src_off = 0;

  for (std::size_t i = 0; i < numel; ++i) {
    dst[i] = src[src_off];

    for (std::size_t d = rank; d-- > 0;) {
      ++idx[d];
      src_off += strides[d];
      if (idx[d] < shape[d]) {
        break;
      }
      src_off -= strides[d] * shape[d];
      idx[d] = 0;
    }
  }
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
} // namespace

namespace functional {

// NOTE: CONSTRUCTOR START ----------------------
Tensor::Tensor(std::vector<std::size_t> shape)
    : shape_(std::move(shape)), strides_(shape_.size()), numel_(1),
      data_(nullptr) {

  strides_ = compute_strides_(shape_, numel_);
  data_ = std::shared_ptr<float[]>(new float[numel_]());
}

Tensor::Tensor(std::shared_ptr<float[]> data, std::vector<std::size_t> shape,
               std::vector<std::size_t> strides, std::size_t numel) noexcept
    : shape_(std::move(shape)), strides_(std::move(strides)), numel_(numel),
      data_(std::move(data)) {}

// NOTE: CONSTRUCTOR END ----------------------

bool Tensor::is_contiguous() const {
  if (numel_ == 0) {
    return true;
  }

  std::size_t expected = 1;
  for (std::size_t d = shape_.size(); d-- > 0;) {
    if (shape_[d] != 1 && strides_[d] != expected) {
      return false;
    }
    expected *= shape_[d];
  }

  return true;
}

// IMP: RULE OF 5 START ----------------------------

Tensor::~Tensor() = default;

Tensor::Tensor(const Tensor &other)
    : shape_(other.shape_), strides_(other.strides_), numel_(other.numel_),
      data_(new float[other.numel_]) {
  std::size_t n = 0;
  strides_ = compute_strides_(shape_, n);

  if (other.is_contiguous()) {
    std::copy(other.data_.get(), other.data_.get() + numel_, data_.get());
  } else {
    gather_logical(other.data_.get(), other.shape_, other.strides_, data_.get(),
                   numel_);
  }
}

Tensor &Tensor::operator=(const Tensor &other) {

  if (this != &other) {
    Tensor tmp(other);
    *this = std::move(tmp);
  }

  return *this;
}

Tensor::Tensor(Tensor &&other) noexcept
    : shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
      numel_(other.numel_), data_(std::move(other.data_)) {
  other.numel_ = 0;
  other.shape_.clear();
  other.strides_.clear();
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
  if (this != &other) {
    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    numel_ = other.numel_;
    data_ = std::move(other.data_);
    other.numel_ = 0;
    other.shape_.clear();
    other.strides_.clear();
  }

  return *this;
}

// IMP: RULE OF 5 END -----------------------------

// NOTE: SHAPE MANIPULATION START -----------------------------------

void Tensor::reshape(std::vector<std::size_t> new_shape) {

  if (!is_contiguous()) {
    throw error::ShapeError("Tensor::reshape",
                            "a contiguous tensor (call clone() first)", shape_);
  }
  std::size_t new_numel = 1;
  std::vector<std::size_t> new_strides = compute_strides_(new_shape, new_numel);

  if (new_numel != numel_) {
    throw error::ShapeError(
        "Tensor::reshape",
        error::make_message("a shape with ", numel_, " elements"), new_shape,
        error::make_message("Element Count mismatch: requested ", new_numel,
                            " elements, but Tensor has ", numel_, " elements"));
  }
  shape_ = std::move(new_shape);
  strides_ = std::move(new_strides);
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

  std::copy(data_.get(), data_.get() + numel_, new_tensor.data_.get());

  return new_tensor;
}

Tensor Tensor::unsqueeze_copy(int &dim) const {
  const int d = normalize_dim(dim, rank());
  std::vector<std::size_t> new_shape = shape_;

  new_shape.insert(new_shape.begin() + d, 1);
  Tensor new_tensor(new_shape);

  std::copy(data_.get(), data_.get() + numel_, new_tensor.data_.get());

  return new_tensor;
}

Tensor Tensor::view(std::vector<std::size_t> shape) const {
  if (!is_contiguous()) {
    throw error::ShapeError("Tensor::view",
                            "a contiguous tensor (call clone() first)", shape_);
  }
  std::size_t new_numel = 1;
  const std::vector<std::size_t> new_stride =
      compute_strides_(shape, new_numel);

  if (new_numel != numel_) {
    throw error::ShapeError(
        "Tensor::view",
        error::make_message("a shape with ", numel_, " elements"), shape,
        error::make_message("Element Count mismatch: requested ", new_numel,
                            " elements, but Tensor has ", numel_, "elements"));
  }

  Tensor new_tensor(data_, std::move(shape), new_stride, new_numel);

  return new_tensor;
}

Tensor Tensor::permute(std::vector<std::size_t> perm) const {
  const std::size_t r = rank();

  if (perm.size() != r) {
    throw error::ShapeError(
        "Tensor::permute",
        "given perm length must have be equal to the Tensor::rank", perm,
        error::make_message("current tensor rank ", r));
  }

  std::vector<bool> bitmap(r, false);

  for (std::size_t p : perm) {
    if (!bound_check(p, static_cast<std::size_t>(0), r - 1)) {
      throw error::IndexError(
          error::make_message(
              "Tensor::permute: given index should be between 0 and ", r - 1,
              " (rank - 1)"),
          p);
    }

    if (bitmap[p]) {
      throw error::ShapeError("Tensor::permute",
                              "permutation indices must be unique", perm);
    }
    bitmap[p] = true;
  }

  std::vector<std::size_t> new_shape = shape_;
  std::vector<std::size_t> new_stride = strides_;

  for (std::size_t i = 0; i < r; i++) {
    new_shape[i] = shape_[perm[i]];
    new_stride[i] = strides_[perm[i]];
  }

  Tensor new_tensor(data_, std::move(new_shape), std::move(new_stride), numel_);

  return new_tensor;
}

Tensor Tensor::transpose(int dim0, int dim1) const {
  int _rank = static_cast<int>(rank());
  if (_rank < 2) {
    throw error::ShapeError("Tensor::transpose",
                            "you need rank >= 2 for transpose", shape_);
  }
  if (!bound_check(dim0, -_rank, _rank - 1) ||
      !bound_check(dim1, -_rank, _rank - 1)) {
    throw error::IndexError("dim0 or dim1 is out of bounds, dim0", dim0,
                            error::make_message("dim1 got index: ", dim1));
  }

  if (dim0 < 0)
    dim0 += _rank;

  if (dim1 < 0)
    dim1 += _rank;

  std::vector<size_t> perm(rank());

  std::iota(perm.begin(), perm.end(), std::size_t{0});
  std::swap(perm[static_cast<std::size_t>(dim0)],
            perm[static_cast<std::size_t>(dim1)]);

  return permute(std::move(perm));
}

Tensor Tensor::broadcast_to(std::vector<std::size_t> shape) const {
  const std::size_t target_rank = shape.size();
  const std::size_t source_rank = rank();

  if (!is_broadcast_possible(shape_, shape)) {
    throw error::ShapeError(
        "Tensor::broadcast_to",
        "the tensor cannot be broadcasted to lower dimensions", shape,
        error::make_message("current tensor shape: ",
                            error::format_shape(shape_)));
  }

  std::vector<std::size_t> new_stride(target_rank, 0);

  const std::size_t offset = target_rank - source_rank;

  for (std::size_t i = 0; i < source_rank; i++) {
    const std::size_t src_dim = shape_[i];
    const std::size_t target_dim = shape[offset + i];

    if (src_dim == target_dim) {
      new_stride[offset + i] = strides_[i];
    } else if (src_dim == 1) {
      new_stride[offset + i] = 0;
    } else {
      throw error::ShapeError(
          "Tensor::broadcast_to", "got incompatible shapes", shape,
          error::make_message(
              "got incompatible shapes, source dim (current tensor) = ",
              src_dim, " given shape dim = ", target_dim));
    }
  }

  std::size_t new_numel = 1;

  for (std::size_t dim : shape) {
    if (dim != 0 && new_numel > std::numeric_limits<std::size_t>::max() / dim) {
      throw std::overflow_error("Tensor::broadcast_to: Tensor size overflow");
    }

    new_numel *= dim;
  }

  return Tensor(data_, std::move(shape), std::move(new_stride), new_numel);
}

Tensor Tensor::broadcast_like(const Tensor &t) const {
  return broadcast_to(t.shape());
}

// NOTE: SHAPE MANIPULATION END -----------------------------------

// NOTE: CASTING START ------------------------------

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

// NOTE: CASTING END ------------------------------

// NOTE: OPERATIONS START -----------------------

Tensor Tensor::operator+(const Tensor &other) const {

  std::vector<std::size_t> other_shape = other.shape();
  std::vector<std::size_t> other_stride = other.strides();

  const std::vector<std::size_t> out_shape =
      Tensor::broadcast_shape(shape_, other_shape);

  const std::vector<std::size_t> a_strides =
      Tensor::broadcast_strides(shape_, strides_, out_shape);
  const std::vector<std::size_t> b_strides =
      Tensor::broadcast_strides(other_shape, other_stride, out_shape);

  Tensor output(out_shape);

  const std::size_t out_rank = output.rank();
  const std::size_t out_numel = output.numel();

  const float *a_data = data_.get();
  const float *b_data = other.data();

  float *out_data = output.data();

  for (std::size_t l = 0; l < out_numel; l++) {
    std::size_t remainder = l;
    std::size_t a_offset = 0;
    std::size_t b_offset = 0;

    for (std::size_t d = out_rank; d-- > 0;) {
      const std::size_t coord = remainder % out_shape[d];
      remainder /= out_shape[d];

      a_offset += coord * a_strides[d];
      b_offset += coord * b_strides[d];
    }

    out_data[l] = a_data[a_offset] + b_data[b_offset];
  }

  return output;
}

Tensor Tensor::operator+(const float &other) const {
  Tensor new_tensor(shape_);
  const float *curr_data = data_.get();
  float *new_data = new_tensor.data();

  for (std::size_t i = 0; i < numel_; i++) {
    new_data[i] = curr_data[i] + other;
  }

  return new_tensor;
}

Tensor Tensor::operator*(const float &other) const {
  Tensor new_tensor(shape_);
  const float *curr_data = data_.get();

  float *new_data = new_tensor.data();

  for (std::size_t i = 0; i < numel_; i++) {
    new_data[i] = curr_data[i] * other;
  }

  return new_tensor;
}

Tensor Tensor::operator/(const float &other) const {
  return this * (1.0f / other);
}

void Tensor::add(const Tensor &other) {
  const std::vector<std::size_t> out_shape =
      broadcast_shape(shape_, other.shape_);

  if (out_shape != shape_) {
    throw error::ShapeError(
        "Tensor::add", "In place addition cannot change the tensor's shape",
        out_shape,
        error::make_message("current shape: ", error::format_shape(shape_)));
  }

  const std::vector<std::size_t> a_strides =
      broadcast_strides(shape_, strides_, out_shape);
  const std::vector<std::size_t> b_strides =
      broadcast_strides(other.shape_, other.strides_, out_shape);

  float *a_data = data_.get();
  const float *b_data = other.data_.get();

  for (std::size_t l = 0; l < numel_; l++) {
    std::size_t remainder = l;
    std::size_t a_offset = 0;
    std::size_t b_offset = 0;

    for (std::size_t d = shape_.size(); d-- > 0;) {
      const std::size_t coord = remainder % out_shape[d];
      remainder /= out_shape[d];

      a_offset += coord * a_strides[d];
      b_offset += coord * b_strides[d];
    }

    a_data[a_offset] += b_data[b_offset];
  }
}

void Tensor::add(const float &other) {
  float *data = data_.get();
  for (std::size_t i = 0; i < numel_; i++) {
    data[i] += other;
  }
}

void Tensor::multiply(const float &other) {
  float *data = data_.get();

  for (std::size_t i = 0; i < numel_; i++) {
    data[i] *= other;
  }
}

void Tensor::divide(const float &other) {
  this->multiply((1.0f / other));
}

Tensor Tensor::sum(int axis, bool keepdims) const {
  const int rank_ = static_cast<int>(rank());

  if (!bound_check(axis, -rank_, rank_)) {
    throw error::IndexError("Tensor::sum: axis is out of bounds", axis,
                            error::make_message("axis needs to be between ( ",
                                                -rank_, ", ", rank_, ")"));
  }

  std::size_t normalized_dim = axis < 0 ? axis + rank_ : axis;

  std::vector<std::size_t> out_shape =
      Tensor::reduction_size(shape_, axis, keepdims);
  Tensor new_tensor(out_shape);

  std::vector<std::size_t> out_stride = new_tensor.strides_;

  new_tensor.fill(0.0f);

  float *new_data = new_tensor.data();
  // std::cerr << "input shape: " << error::format_shape(shape_)
  //           << "\naxis: " << axis
  //           << "\noutput shape: " << error::format_shape(out_shape)
  //           << "\noutput strides: " << error::format_shape(out_stride) <<
  //           '\n';

  Tensor::for_each(
      shape_, strides_, out_stride, normalized_dim, keepdims,
      [&](std::size_t input_offset, std::size_t output_offset, std::size_t &) {
        new_data[output_offset] += data_[input_offset];
      });

  return new_tensor;
}

// NOTE: OPERATIONS END

// NOTE: MISC START --------------------

void Tensor::fill(float value) {
  std::fill(data_.get(), data_.get() + numel_, value);
}

Tensor Tensor::clone() const { return Tensor(*this); }

std::vector<std::size_t>
Tensor::broadcast_strides(const std::vector<std::size_t> &shape,
                          const std::vector<std::size_t> &input_strides,
                          const std::vector<std::size_t> &output_shape) {
  const std::size_t input_rank = shape.size();
  const std::size_t output_rank = output_shape.size();

  std::vector<std::size_t> effective_strides(output_rank, 0);

  const std::size_t rank_offset = output_rank - input_rank;

  for (std::size_t d = 0; d < input_rank; d++) {
    const std::size_t out_d = d + rank_offset;

    if (shape[d] == output_shape[out_d]) {
      effective_strides[out_d] = input_strides[d];
    } else if (shape[d] == 1) {
      effective_strides[out_d] = 0;
    } else {
      throw error::ShapeError(
          "broadcast_strides", "Input shape is incompatible with output shape",
          shape,
          error::make_message("expected shape: ",
                              error::format_shape(output_shape)));
    }
  }

  return effective_strides;
}

std::vector<std::size_t>
Tensor::reduction_size(const std::vector<std::size_t> &shape, int axis,
                       bool keepdims) {
  const int rank_ = static_cast<int>(shape.size());

  if (rank_ == 0) {
    throw error::ShapeError("Tensor::(reducing axis)",
                            "cannot reduce a tensor with single value", shape);
  }

  if (!bound_check(axis, -rank_, rank_)) {
    throw error::IndexError(
        "Tensor::(reducing axis): given axis value exceeds bounds", axis,
        error::make_message("axis needs to be between ( ", -rank_, ", ", rank_,
                            ")"));
  }

  int normalized_axis = axis < 0 ? (axis + rank_) : axis;

  std::vector<std::size_t> new_shape = shape;

  if (keepdims)
    new_shape[normalized_axis] = 1;
  else
    new_shape.erase(new_shape.begin() + normalized_axis);

  return new_shape;
}

std::vector<std::size_t>
Tensor::reduction_size(const std::vector<std::size_t> &shape,
                       std::vector<int> &axes, bool keepdims) {

  const int rank_ = static_cast<int>(shape.size());

  std::vector<bool> seen_axes(shape.size(), false);

  for (std::size_t i = 0; i < axes.size(); ++i) {
    int curr_axis = axes[i];
    if (!bound_check(curr_axis, -rank_, rank_)) {
      throw error::IndexError(
          "Tensor::(reducing axis): given axis value exceeds bounds", curr_axis,
          error::make_message("axis needs to be between ( ", -rank_, ", ",
                              rank_, ")"));
    }

    int normalized_axis = curr_axis < 0 ? (curr_axis + rank_) : curr_axis;

    if (seen_axes[normalized_axis]) {
      throw error::IndexError(
          "Tensor::(reducing axis): duplicate axis found", curr_axis,
          error::make_message(" normalized axis", normalized_axis));
    }

    seen_axes[normalized_axis] = true;
  }

  std::vector<std::size_t> new_shape = shape;

  for (std::size_t d = 0; d < shape.size(); ++d) {
    if (!seen_axes[d])
      continue;

    if (keepdims)
      new_shape[d] = 1;
    else
      new_shape.erase(new_shape.begin() + d);
  }

  return new_shape;
}

// std::size_t Tensor::unravel_index(std::size_t linear,
//                                   const std::vector<std::size_t> &shape) {
//
//   const std::size_t remainder = linear;
//
//   for (std::size_t d = shape.size(); d-- < 0;) {
//
//   }
// }

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

bool Tensor::is_broadcast_possible(
    const std::vector<std::size_t> &got_shape,
    const std::vector<std::size_t> &target_shape) noexcept {
  const std::size_t gs_rank = got_shape.size();
  const std::size_t ts_rank = target_shape.size();

  if (gs_rank > ts_rank) {
    return false;
  }

  std::vector<std::size_t> output_shape(ts_rank);

  for (std::size_t i = 0; i < ts_rank; ++i) {
    const std::size_t dim =
        (i < ts_rank - gs_rank) ? 1 : got_shape[i - ts_rank + gs_rank];
    const std::size_t dim_t = target_shape[i];

    if (dim_t != dim && dim != 1) {
      return false;
    }
  }

  return true;
}

std::vector<std::size_t>
Tensor::broadcast_shape(const std::vector<std::size_t> &shape_a,
                        const std::vector<std::size_t> &shape_b) {

  const std::size_t a_rank = shape_a.size();
  const std::size_t b_rank = shape_b.size();
  const std::size_t out_rank = std::max(a_rank, b_rank);

  std::vector<std::size_t> output_shape(out_rank);

  for (std::size_t i = 0; i < out_rank; ++i) {
    const std::size_t dim_a =
        (i < out_rank - a_rank) ? 1 : shape_a[i - out_rank + a_rank];
    const std::size_t dim_b =
        (i < out_rank - b_rank) ? 1 : shape_b[i - out_rank + b_rank];

    if (dim_a == dim_b) {

      output_shape[i] = dim_a;
    } else if (dim_a == 1) {
      output_shape[i] = dim_b;
    } else if (dim_b == 1) {
      output_shape[i] = dim_a;
    } else {
      throw error::ShapeError(
          "Tensor::broadcast_shape",
          "Incompatible shapes for broadcasting, shape A", shape_a,
          error::make_message("B Tensor got shape: ",
                              error::format_shape(shape_b)));
    }
  }

  return output_shape;
}

// NOTE: MISC END--------------------
//

void require_contiguous(const Tensor &t, const std::string &operation,
                        const std::string &variable_name) {
  if (!t.is_contiguous()) {
    throw error::ShapeError(
        operation,
        error::make_message(variable_name.empty() ? "Tensor" : variable_name,
                            " needs to be contiguous"),
        t.shape(), "try running clone() first...");
  }
}

Tensor reshape(Tensor &tensor, std::vector<std::size_t> &shape) {
  Tensor t_diff = tensor;

  t_diff.reshape(shape);

  return t_diff;
}

} // namespace functional
