#include <core/errors.h>
#include <core/tensor.h>
#include <cstddef>
#include <ops/utils.h>

using core::Tensor;
namespace error = core::error;

namespace ops::utils {

std::size_t batch_offset(std::size_t flat_batch,
                         const std::vector<std::size_t> &shape,
                         const std::vector<std::size_t> &strides) {

  const std::size_t batch_rank = shape.size() - 2;

  std::size_t offset = 0;

  for (std::size_t i = batch_rank; i-- > 0;) {
    const std::size_t index = flat_batch % shape[i];
    flat_batch /= shape[i];

    offset += index * strides[i];
  }

  return offset;
}

// calculate the offset
std::size_t batch_offset(const std::vector<std::size_t> &batch_index,
                         const std::size_t &batch_rank,
                         const std::size_t &input_batch_rank,
                         const std::vector<std::size_t> &input_shape,
                         const std::vector<std::size_t> &input_stride) {
  std::size_t offset = 0;

  // going through each given inputs batch rank
  for (std::size_t d = 0; d < input_batch_rank; d++) {

    // the current out_dimension to hit
    // for eg. given output batch_rank of 3 of index (0, 1, 1)
    // and input batch rank of 2 of input shape (3, 2, 2, 2) and therefore
    // stride ( 8, 4, 2, 1) current loop of 2 i.e d = 1
    // then out_dimension would be = 3 - 2 - 1 = 0
    // the offset = (1  *  8) + (0 * 2) => 8
    // we skip if input_shape[d] = 1, for broadcasting
    if (input_shape[d] != 1) {
      const std::size_t out_d = batch_rank - input_batch_rank + d;
      offset += batch_index[out_d] * input_stride[d];
    }
  }

  return offset;
}

std::vector<std::size_t> matmul_shape(const Tensor &A, const Tensor &B,
                                      bool transpose_a, bool transpose_b) {
  return matmul_shape(A.shape(), B.shape(), transpose_a, transpose_b);
}
std::vector<std::size_t> matmul_shape(const std::vector<std::size_t> shape_a,
                                      const std::vector<std::size_t> shape_b,
                                      bool transpose_a, bool transpose_b) {
  if (shape_b.empty() || shape_a.empty()) {
    throw error::ShapeError(
        "matmul:", "scaler multiplcation is not allowed!", shape_a,
        error::make_message("B shape: ", error::format_shape(shape_b)));
  }

  const std::size_t rank_a = shape_a.size();
  const std::size_t rank_b = shape_b.size();

  const bool a_vector = rank_a == 1;
  const bool b_vector = rank_b == 1;

  std::vector<std::size_t> a = shape_a;
  std::vector<std::size_t> b = shape_b;

  if (a_vector) {
    // {1, a}
    a.insert(a.begin(), static_cast<std::size_t>(1));
  }

  if (b_vector) {
    // {b, 1}
    b.push_back(1);
  }

  const std::size_t effective_rank_a = a.size();
  const std::size_t effective_rank_b = b.size();

  const std::size_t M = transpose_a ? a.back() : a[effective_rank_a - 2];
  const std::size_t k_a = transpose_a ? a[effective_rank_a - 2] : a.back();
  const std::size_t k_b = transpose_b ? b.back() : b[effective_rank_b - 2];
  const std::size_t N = transpose_b ? b[effective_rank_b - 2] : b.back();

  if (k_a != k_b) {
    throw error::ShapeError(
        "matmul",
        "A and B tensors have incompatible shapes for matrix multiplication",
        shape_a,
        error::make_message("found B shape: ", error::format_shape(shape_b)));
  }

  std::vector<std::size_t> batch_a(a.begin(), a.end() - 2);
  std::vector<std::size_t> batch_b(b.begin(), b.end() - 2);

  std::vector<std::size_t> batch_out =
      Tensor::broadcast_shape(batch_a, batch_b);

  batch_out.push_back(M);
  batch_out.push_back(N);

  if (a_vector) {
    batch_out.erase(batch_out.end() - 2);
  }

  if (b_vector) {
    batch_out.pop_back();
  }

  return batch_out;
}

} // namespace ops::utils
