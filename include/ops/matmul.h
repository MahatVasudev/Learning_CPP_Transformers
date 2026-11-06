#ifndef OPS_MATMUL

#define OPS_MATMUL

#include <core/tensor.h>

using core::Tensor;

namespace ops {

void matmul(const Tensor &A, const Tensor &B, Tensor &C,
            bool transpose_a = false, bool transpose_b = false,
            bool accumulate = false);

Tensor matmul(const Tensor &A, const Tensor &B, bool transpose_a = false,
              bool transpose_b = false, bool accumulate = false);

// deprecated
void matmul_backward(const Tensor &A, const Tensor &B, const Tensor &dY,
                     Tensor &dA, Tensor &dB, bool accumulate = true);

// NOTE: Generalized Matrix Multiplication
void matmul_batched(const Tensor &A, const Tensor &B, Tensor &C,
                    bool transpose_a, bool transpose_b, bool accumulate);

// NOTE: Specific version of matmul_batched, when no broadcasting is needed
void bmm(const Tensor &A, const Tensor &B, Tensor &C);

// NOTE: Atomic operation for matrix multiplication of rank 2
void gemm(const float *A, const float *B, float *C, std::size_t M,
          std::size_t K, std::size_t N, std::size_t a_stride0,
          std::size_t a_stride1, std::size_t b_stride0, std::size_t b_stride1,
          std::size_t c_stride0, std::size_t c_stride1, bool transA,
          bool transB, bool accumulate);

} // namespace ops

#endif // !OPS_MATMUL
