#include "functional/gradcheck.h"
#include "functional/ops.h"
#include "functional/tensor.h"
#include "tests/tests.h"

#include "tests/expect.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

using namespace tests;
using namespace functional;

static void fill_random(Tensor &t, std::mt19937 &rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t n = 0; n < t.numel(); ++n)
    t.data()[n] = dist(rng);
}

static bool close(const Tensor &a, const Tensor &b, float tol = 1e-5f) {
  if (a.shape() != b.shape())
    return false;
  for (std::size_t n = 0; n < a.numel(); ++n)
    if (std::abs(a.data()[n] - b.data()[n]) > tol)
      return false;
  return true;
}

static Tensor transposed(const Tensor &t) {
  Tensor out({t.shape()[1], t.shape()[0]});
  for (std::size_t i = 0; i < t.shape()[0]; ++i)
    for (std::size_t j = 0; j < t.shape()[1]; ++j)
      out(j, i) = t(i, j);
  return out;
}

// 1. Forward against a hand-computed answer.
static void test_forward_hand() {
  Tensor A({2, 2}), B({2, 2}), C({2, 2});
  A(0, 0) = 1;
  A(0, 1) = 2;
  A(1, 0) = 3;
  A(1, 1) = 4;
  B(0, 0) = 5;
  B(0, 1) = 6;
  B(1, 0) = 7;
  B(1, 1) = 8;
  matmul(A, B, C);
  EXPECT(C(0, 0) == 19 && C(0, 1) == 22);
  EXPECT(C(1, 0) == 43 && C(1, 1) == 50);
  std::cout << "forward (hand-checked): ok\n";
}

// 2. Transpose flags: passing a pre-transposed matrix with the flag set
//    must give the same answer as the plain product. Non-square on purpose.
static void test_forward_transpose_flags() {
  std::mt19937 rng(1);
  Tensor A({3, 4}), B({4, 5});
  fill_random(A, rng);
  fill_random(B, rng);

  Tensor expected({3, 5});
  matmul(A, B, expected);

  Tensor At = transposed(A), Bt = transposed(B);
  Tensor C({3, 5});

  matmul(At, B, C, /*transA=*/true, /*transB=*/false);
  EXPECT(close(C, expected));
  matmul(A, Bt, C, false, true);
  EXPECT(close(C, expected));
  matmul(At, Bt, C, true, true);
  EXPECT(close(C, expected));
  std::cout << "forward (transpose flags): ok\n";
}

// 3. Accumulate adds to C instead of overwriting.
static void test_accumulate() {
  std::mt19937 rng(2);
  Tensor A({3, 4}), B({4, 5});
  fill_random(A, rng);
  fill_random(B, rng);

  Tensor once({3, 5}), twice({3, 5});
  matmul(A, B, once);
  matmul(A, B, twice);
  matmul(A, B, twice, false, false, /*accumulate=*/true);
  for (std::size_t n = 0; n < once.numel(); ++n)
    EXPECT(std::abs(twice.data()[n] - 2.0f * once.data()[n]) < 1e-5f);
  std::cout << "accumulate: ok\n";
}

// 4. Backward against the hand-computed values from earlier.
static void test_backward_hand() {
  Tensor A({2, 2}), B({2, 2}), dY({2, 2}), dA({2, 2}), dB({2, 2});
  A(0, 0) = 1;
  A(0, 1) = 2;
  A(1, 0) = 3;
  A(1, 1) = 4;
  B(0, 0) = 5;
  B(0, 1) = 6;
  B(1, 0) = 7;
  B(1, 1) = 8;
  dY.fill(1.0f);
  matmul_backward(A, B, dY, dA, dB, /*accumulate=*/false);
  EXPECT(dA(0, 0) == 11 && dA(0, 1) == 15);
  EXPECT(dA(1, 0) == 11 && dA(1, 1) == 15);
  EXPECT(dB(0, 0) == 4 && dB(0, 1) == 4);
  EXPECT(dB(1, 0) == 6 && dB(1, 1) == 6);
  std::cout << "backward (hand-checked): ok\n";
}

// 5. Gradient check, plus a self-test that the checker can actually fail.
static void test_gradcheck() {
  std::mt19937 rng(3);
  const std::size_t M = 3, K = 4, N = 5;
  Tensor A({M, K}), B({K, N}), Y({M, N}), R({M, N});
  fill_random(A, rng);
  fill_random(B, rng);
  fill_random(R, rng);

  // Scalar loss L = sum(Y * R), accumulated in double. Then dL/dY == R.
  auto loss = [&]() {
    matmul(A, B, Y);
    double L = 0.0;
    for (std::size_t n = 0; n < Y.numel(); ++n)
      L += static_cast<double>(Y.data()[n]) * R.data()[n];
    return L;
  };

  Tensor dA({M, K}), dB({K, N}); // zero-initialized
  matmul_backward(A, B, /*dY=*/R, dA, dB, /*accumulate=*/false);

  GradcheckResult ra = check_gradient(A, dA, loss);
  GradcheckResult rb = check_gradient(B, dB, loss);
  std::cout << "gradcheck A: max rel err " << ra.max_rel_error << "\n";
  std::cout << "gradcheck B: max rel err " << rb.max_rel_error << "\n";
  EXPECT(ra.passed);
  EXPECT(rb.passed);

  // Self-test: a deliberately wrong gradient (10% too large) must FAIL.
  Tensor dA_bad = dA.clone();
  for (std::size_t n = 0; n < dA_bad.numel(); ++n)
    dA_bad.data()[n] *= 1.1f;
  GradcheckResult bad = check_gradient(A, dA_bad, loss);
  std::cout << "gradcheck on broken dA: max rel err " << bad.max_rel_error
            << " (expected to fail)\n";
  EXPECT(!bad.passed);

  // Self-test: NaN in the analytic gradient must also FAIL.
  Tensor dA_nan = dA.clone();
  dA_nan.data()[0] = std::nanf("");
  EXPECT(!check_gradient(A, dA_nan, loss).passed);

  std::cout << "gradcheck (incl. self-tests): ok\n";
}

void tests::run_matmul() {
  test_forward_hand();
  test_forward_transpose_flags();
  test_accumulate();
  test_backward_hand();
  test_gradcheck();
  std::cout << "all matmul tests passed\n";
}
