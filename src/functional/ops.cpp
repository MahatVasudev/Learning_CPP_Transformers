#include "functional/ops.h"
#include <cassert>
#include <cstddef>
#include <functional>

namespace functional {

void matmul(const Tensor &A, const Tensor &B, Tensor &C, bool transA,
            bool transB, bool accumulate) {

  std::size_t M = transA ? A.shape()[1] : A.shape()[0];
  std::size_t K = transA ? A.shape()[0] : A.shape()[1];

  std::size_t K2 = transB ? B.shape()[1] : B.shape()[0];
  std::size_t N = transB ? B.shape()[0] : B.shape()[1];

  assert(K == K2);
  assert(C.shape()[0] == M);
  assert(C.shape()[1] == N);

  std::size_t lda = transA ? M : K;
  std::size_t ldb = transB ? K : N;
  std::size_t ldc = N;

  gemm(A.data(), B.data(), C.data(), M, K, N, lda, ldb, ldc, transA, transB,
       accumulate);
}

void matmul_backward(const Tensor &A, const Tensor &B, const Tensor &dY,
                     Tensor &dA, Tensor &dB, bool accumulate) {
  assert(dY.rank() == 2 && dA.rank() == 2 && dB.rank() == 2);
  assert(dA.shape() == A.shape());
  assert(dB.shape() == B.shape());
  assert(dY.shape()[0] == A.shape()[0] && dY.shape()[1] == B.shape()[1]);

  matmul(dY, B, dA, false, true, accumulate);

  matmul(A, dY, dB, true, false, accumulate);
}

static bool overlaps(const float *p, std::size_t p_len, const float *q,
                     const std::size_t q_len) {
  std::less<const float *> lt;
  return lt(p, q + q_len) && lt(q, p + p_len);
}

void gemm(const float *A, const float *B, float *C, std::size_t M,
          std::size_t K, std::size_t N, std::size_t lda, std::size_t ldb,
          std::size_t ldc, bool transA, bool transB, bool accumulate) {
  assert(A != nullptr && B != nullptr && C != nullptr);
  assert(lda >= (transA ? M : K));
  assert(ldb >= (transB ? K : N));
  assert(ldc >= N);
  const std::size_t a_rows = transA ? K : M, a_cols = transA ? M : K;
  const std::size_t b_rows = transB ? N : K, b_cols = transB ? K : N;
  const std::size_t a_len = (a_rows - 1) * lda + a_cols;
  const std::size_t b_len = (b_rows - 1) * ldb + b_cols;
  const std::size_t c_len = (M - 1) * ldc + N;

  assert(!overlaps(C, c_len, A, a_len));
  assert(!overlaps(C, c_len, B, b_len));

  if (!accumulate) {
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        C[i * ldc + j] = 0.0f;
      }
    }
  }

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t k = 0; k < K; ++k) {
      float a = transA ? A[k * lda + i] : A[i * lda + k];

      for (std::size_t j = 0; j < N; ++j) {
        float b = transB ? B[j * ldb + k] : B[k * ldb + j];

        C[i * ldc + j] += a * b;
      }
    }
  }
}

} // namespace functional
