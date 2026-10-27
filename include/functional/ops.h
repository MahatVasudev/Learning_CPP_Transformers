#ifndef OPS_FUNCTIONAL_H

#define OPS_FUNCTIONAL_H

#include "functional/tensor.h"
namespace functional {

void matmul(const Tensor &A, const Tensor &B, Tensor &C,
            bool transpose_a = false, bool transpose_b = false,
            bool accumulate = false);

Tensor matmul(const Tensor &A, const Tensor &B, bool transpose_a = false,
              bool transpose_b = false, bool accumulate = false);

void matmul_backward(const Tensor &A, const Tensor &B, const Tensor &dY,
                     Tensor &dA, Tensor &dB, bool accumulate = true);

std::vector<std::size_t> matmul_shape(const std::vector<std::size_t> shape_a,
                                      const std::vector<std::size_t> shape_b,
                                      bool transpose_a = false,
                                      bool transpose_b = false);

std::vector<std::size_t> matmul_shape(const Tensor &A, const Tensor &B,
                                      bool transpose_a = false,
                                      bool tranpose_b = false);

// NOTE: Generalized Matrix Multiplication
void matmul_batched(const Tensor &A, const Tensor &B, Tensor &C,
                    bool transpose_a, bool transpose_b, bool accumulate);

// NOTE: Specific version of matmul_batched, when no broadcasting is needed
void bmm(const Tensor &A, const Tensor &B, Tensor &C);

// NOTE: Atomic operation for 2x2 matrix multiplication
void gemm(const float *A, const float *B, float *C, std::size_t M,
          std::size_t K, std::size_t N, std::size_t a_stride0,
          std::size_t a_stride1, std::size_t b_stride0, std::size_t b_stride1,
          std::size_t c_stride0, std::size_t c_stride1, bool transA,
          bool transB, bool accumulate);

bool overlaps(const float *p, std::size_t p_len, const float *q,
              const std::size_t q_len);

// batch_offset, calculating batch offset of tensor Tensor
std::size_t batch_offset(std::size_t flat_batch,
                         const std::vector<std::size_t> &shape,
                         const std::vector<std::size_t> &strides);

// batch_offset, calculating offset on a broad-cast Tensor
std::size_t batch_offset(const std::vector<std::size_t> &batch_index,
                         const std::size_t &batch_rank,
                         const std::size_t &input_batch_rank,
                         const std::vector<std::size_t> &input_shape,
                         const std::vector<std::size_t> &input_stride);
} // namespace functional

#endif // !OPS_FUNCTIONAL_H
