#include "ops/ops.h"
#include "core/errors.h"
#include <cassert>
#include <cstddef>
#include <functional>
#include <vector>

using core::Tensor;
namespace error = core::error;

namespace ops {

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

void matmul(const Tensor &A, const Tensor &B, Tensor &C, bool transpose_a,
            bool transpose_b, bool accumulate) {
  matmul_batched(A, B, C, transpose_a, transpose_b, accumulate);
}

Tensor matmul(const Tensor &A, const Tensor &B, bool transpose_a,
              bool transpose_b, bool accumulate) {

  std::vector<std::size_t> shape = matmul_shape(A, B, transpose_a, transpose_b);

  Tensor C(shape);

  matmul_batched(A, B, C, transpose_a, transpose_b, accumulate);

  return C;
}

void matmul_backward(const Tensor &A, const Tensor &B, const Tensor &dY,
                     Tensor &dA, Tensor &dB, bool accumulate) {
  assert(dA.shape() == A.shape());
  assert(dB.shape() == B.shape());
  assert(dY.shape()[0] == A.shape()[0] && dY.shape()[1] == B.shape()[1]);

  matmul(dY, B, dA, false, true, accumulate);

  matmul(A, dY, dB, true, false, accumulate);
}

bool overlaps(const float *p, std::size_t p_len, const float *q,
              const std::size_t q_len) {
  std::less<const float *> lt;
  return lt(p, q + q_len) && lt(q, p + p_len);
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

void matmul_batched(const Tensor &A, const Tensor &B, Tensor &C,
                    bool transpose_a, bool transpose_b, bool accumulate) {

  // Get Shapes of the tensors
  std::vector<std::size_t> a_shape = A.shape();
  std::vector<std::size_t> a_stride = A.strides();
  std::vector<std::size_t> b_shape = B.shape();
  std::vector<std::size_t> b_stride = B.strides();
  std::vector<std::size_t> c_shape = C.shape();
  std::vector<std::size_t> c_stride = C.strides();
  std::vector<std::size_t> c_strides_effective = c_stride;

  // check whether A or B is a vector
  const bool a_vector = a_shape.size() == 1;
  const bool b_vector = b_shape.size() == 1;

  // if a is a vector then make it shape (1, n)
  if (a_vector) {
    a_shape.insert(a_shape.begin(), 1);
    a_stride.insert(a_stride.begin(), 1);
    c_strides_effective.insert(c_stride.end() - 1, 0);
  }

  // if b is a vector then make it shape (n, 1)
  if (b_vector) {
    b_shape.push_back(1);
    b_stride.push_back(0);
    c_strides_effective.push_back(0);
  }

  // calculate the final shape after the matmul of A and B
  std::vector<std::size_t> final_size =
      matmul_shape(a_shape, b_shape, transpose_a, transpose_b);

  // std::cerr << "A: " << error::format_shape(A.shape())
  //           << "\nB: " << error::format_shape(B.shape())
  //           << "\nC: " << error::format_shape(C.shape())
  //           << "\nExpected: " << error::format_shape(final_size) << '\n';
  if (c_shape != final_size) {
    throw error::ShapeError(
        "matumul", "given output shape is not correct", c_shape,
        error::make_message(
            "expected last two value of shape to be of the given shape ",
            error::format_shape(final_size)));
  }

  // get rank of the input tensors
  const std::size_t a_rank = a_shape.size();
  const std::size_t b_rank = b_shape.size();
  // std::size_t c_rank = c_shape.size();

  // fetching the rows and columns through which we will iterate
  // given a Tensor A and B with shape (...$b, $1, $2) and (...$b, $j, $k)
  // and the resulting Tensor C from A * B with shape (...$b, M, N)
  // M dimension would be from Tensor A $2 or $1 if transposed
  // N dimension would be from Tensor B $1 or $2 if transposed
  // K dimension needs to be common between A or B which would be the non M or N
  // dimension
  const std::size_t M = transpose_a ? a_shape[a_rank - 1] : a_shape[a_rank - 2];
  const std::size_t K = transpose_a ? a_shape[a_rank - 2] : a_shape[a_rank - 1];
  const std::size_t N = transpose_b ? b_shape[b_rank - 2] : b_shape[b_rank - 1];

  // keeping tracks of batches count
  // for eg. Given shape of resulting tensor C as (3, 2, 2, 2)
  // batch shape of C is (3, 2) (leaving the last two dimensions)
  // the count of batches then would be 1 x 3 x 2 which is _6_
  std::size_t batch_count = 1;
  const std::vector<std::size_t> batch_shape(final_size.begin(),
                                             final_size.end() - 2);

  for (std::size_t dim : batch_shape) {
    batch_count *= dim;
  }

  // again taking the previous example of Tensor C with shape (3, 2, 2, 2)
  // the batch rank (or number of batch dimensions) would be _2_
  const std::size_t batch_rank = batch_shape.size();

  // we will keep track of each batches index
  std::vector<std::size_t> batch_index(batch_rank, 0);

  // individual input tensors rank, keeping track for broadcasting
  const std::size_t a_batch_rank = a_rank - 2;
  const std::size_t b_batch_rank = b_rank - 2;

  // going through each batch
  for (std::size_t flat = 0; flat < batch_count; flat++) {
    // think of this as the remaining dimension left to hit
    std::size_t remaining = flat;

    // for all batches, worked from backwards
    for (std::size_t d = batch_rank; d-- > 0;) {
      // keeping track of the current index of the batch
      // so for batch rank of two with batch shape (3, 2)
      // and the first parent loop, i.e. flat = remaining = 0
      // first in loop batch_index[1] = 0 % 2 => 0
      // remaining = 0 / 2
      // second in loop batch_index[0] = 0 % 3 => 0
      // for second parent loop, i.e flat = remaining = 1
      // first in loop batch_index[1] = 1 % 2 => 1
      // remaining = 1 / 2 => 0
      // second in loop batch_index[0] = 0 % 3 => 0
      batch_index[d] = remaining % batch_shape[d];
      remaining /= batch_shape[d];
    }

    // calculate the current offset of elements to consider for
    // matrix multiplication from A and B, this offset represents which 2
    // dimensional data to choose for multiplication given the current index of
    // batch we are in.
    std::size_t a_offset =
        batch_offset(batch_index, batch_rank, a_batch_rank, a_shape, a_stride);
    std::size_t b_offset =
        batch_offset(batch_index, batch_rank, b_batch_rank, b_shape, b_stride);

    // which output 2 dimensional matrix to save output of A and B
    // multiplication
    std::size_t c_offset = 0;
    for (std::size_t d = 0; d < batch_rank; d++) {
      c_offset += batch_index[d] * c_stride[d];
    }

    // moving to the point where the 2 dimensional matrix would start
    float *c_ptr = C.data() + c_offset;

    // performing gemm on the chosen 2 dimensional data
    gemm(A.data() + a_offset, B.data() + b_offset, c_ptr, M, K, N,
         a_stride[a_rank - 2], a_stride[a_rank - 1], b_stride[b_rank - 2],
         b_stride[b_rank - 1], c_strides_effective[batch_rank],
         c_strides_effective[batch_rank + 1], transpose_a, transpose_b,
         accumulate);
  }
}

void bmm(const Tensor &A, const Tensor &B, Tensor &C) {

  const std::size_t a_rank = A.rank();
  const std::size_t b_rank = B.rank();
  if ((a_rank != 2 && a_rank != 3) || (b_rank != 2 && b_rank != 3)) {
    throw error::ShapeError(
        "bemm", "Expected A and B to have rank 2 or 3", A.shape(),
        error::make_message("B shape: ", error::format_shape(B.shape())));
  }

  if (a_rank != 3 && b_rank != 3) {
    throw error::ShapeError(
        "bemm", "At least one operand must have rank 3", A.shape(),
        error::make_message("B shape: ", error::format_shape(B.shape())));
  }

  const std::vector<std::size_t> a_shape = A.shape();
  const std::vector<std::size_t> b_shape = B.shape();
  const bool is_a_small = (a_rank == 2);
  const std::size_t batch_a = is_a_small ? 1 : a_shape[0];
  const std::size_t M = is_a_small ? a_shape[0] : a_shape[1];
  const std::size_t K = is_a_small ? a_shape[1] : a_shape[2];

  const std::size_t batch_b = !is_a_small ? 1 : b_shape[0];
  const std::size_t K_b = !is_a_small ? b_shape[0] : b_shape[1];
  const std::size_t N = !is_a_small ? b_shape[1] : b_shape[2];

  if (batch_a != batch_b) {
    throw error::ShapeError(
        "bemm", "Expected A and B to have the same batch size...", a_shape,
        error::make_message("got B shape: ", error::format_shape(b_shape)));
  }

  if (K != K_b) {
    throw error::ShapeError(
        "bemm", "inner dimensions must match for matrix multiplication",
        a_shape,
        error::make_message("B shape: ", error::format_shape(b_shape)));
  }

  const std::vector<std::size_t> expected_shape = {
      is_a_small ? batch_b : batch_a, M, N};

  if (C.shape() != expected_shape) {
    throw error::ShapeError(
        "bemm", "Output tensor has an incorrect shape", C.shape(),
        error::make_message("Expected shape: ",
                            error::format_shape(expected_shape)));
  }

  const std::vector<std::size_t> &a_strides = A.strides();
  const std::vector<std::size_t> &b_strides = B.strides();
  const std::vector<std::size_t> &c_strides = C.strides();

  const std::size_t a_batch_stride =
      (a_rank == 2 || batch_a == 1) ? 0 : a_strides[0];

  const std::size_t a_stride_m = is_a_small ? a_strides[0] : a_strides[1];
  const std::size_t a_stride_k = is_a_small ? a_strides[1] : a_strides[2];

  const std::size_t b_batch_stride =
      (b_rank == 2 || batch_b == 1) ? 0 : b_strides[0];

  const std::size_t b_stride_k = !is_a_small ? b_strides[0] : b_strides[1];
  const std::size_t b_stride_n = !is_a_small ? b_strides[1] : b_strides[2];

  for (std::size_t i = 0; i < batch_a; i++) {
    const float *a_ptr = A.data() + i * a_batch_stride;
    const float *b_ptr = B.data() + i * b_batch_stride;
    float *c_ptr = C.data() + i * c_strides[0];

    gemm(a_ptr, b_ptr, c_ptr, M, K, N, a_stride_m, a_stride_k, b_stride_k,
         b_stride_n, c_strides[1], c_strides[2], false, false, false);
  }
}

// the main matrix multiplication function,
// is stride aware, and handles transpose of matrix
void gemm(const float *A, const float *B, float *C, std::size_t M,
          std::size_t K, std::size_t N, std::size_t a_stride0,
          std::size_t a_stride1, std::size_t b_stride0, std::size_t b_stride1,
          std::size_t c_stride0, std::size_t c_stride1, bool transA,
          bool transB, bool accumulate) {
  if (M == 0 || N == 0)
    return;

  assert(A != nullptr && B != nullptr && C != nullptr);

  assert(a_stride0 > 0 && a_stride1 > 0);
  assert(b_stride0 > 0 && b_stride1 > 0);
  assert(c_stride0 > 0 && c_stride1 > 0);

  // setting everything to zero, if not accumulate
  // to remove the previous values if any
  if (!accumulate) {
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        C[i * c_stride0 + j * c_stride1] = 0.0f;
      }
    }
  }

  // the basic matrix multiplication operation
  // given matrix A with values [[2, 3, 3], [3, 3, 3]]
  // and matrix B with values [[2, 2], [3, 3], [4, 4]]
  // therefore the shape of A is (2, 3) with strides (3, 1)
  // and shape of B is (3, 2) with strides (2, 1)
  // and resulting shape would be (2, 2) with strides (2, 1)
  for (std::size_t i = 0; i < M; ++i) {
    // on 1st iteration of M, i.e i = 0
    for (std::size_t k = 0; k < K; ++k) {
      // on 1st iteration of K, i.e k = 0
      // a becomes A[0 * 3 + 0 * 1] = 2
      float a = transA ? A[k * a_stride0 + i * a_stride1]
                       : A[i * a_stride0 + k * a_stride1];

      for (std::size_t j = 0; j < N; ++j) {
        // on first iteration of N, i.e. j = 0
        // b becomes B[0 * 2 + 0 * 1] = 2
        float b = transB ? B[j * b_stride0 + k * b_stride1]
                         : B[k * b_stride0 + j * b_stride1];

        // storing c[0] += 2 * 2
        C[i * c_stride0 + j * c_stride1] += a * b;
      }
    }
  }
}

} // namespace ops
