#ifndef OPS_FUNCTIONAL_H

#define OPS_FUNCTIONAL_H

#include "functional/tensor.h"
namespace functional {

void matmul(const Tensor &A, const Tensor &B, Tensor &C, bool transA = false,
            bool transB = false, bool accumulate = false);

void matmul_backward(const Tensor &A, const Tensor &B, const Tensor &dY,
                     Tensor &dA, Tensor &dB, bool accumulate = true);

void gemm(const float *A, const float *B, float *C, std::size_t M,
          std::size_t K, std::size_t N, std::size_t lda, std::size_t ldb,
          std::size_t ldc, bool transA, bool transB, bool accumulate);
} // namespace functional

#endif // !OPS_FUNCTIONAL_H
