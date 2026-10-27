#include "functional/gradcheck.h"
#include "functional/module.h" // adjust if Linear lives in another header
#include "functional/ops.h"
#include "functional/tensor.h"
#include "tests/tests.h"

#include "tests/expect.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

using namespace tests;
using namespace functional;

static void fill_random(Tensor &t, std::mt19937 &rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t n = 0; n < t.numel(); ++n)
    t.data()[n] = dist(rng);
}

static double dot_loss(const Tensor &Y, const Tensor &R) {
  double L = 0.0;
  for (std::size_t n = 0; n < Y.numel(); ++n)
    L += static_cast<double>(Y.data()[n]) * R.data()[n];
  return L;
}

static void set_values(Tensor &t, const std::vector<float> &vals) {
  EXPECT(t.numel() == vals.size());
  for (std::size_t n = 0; n < vals.size(); ++n)
    t.data()[n] = vals[n];
}

static bool near(float a, float b, float tol = 1e-5f) {
  return std::abs(a - b) <= tol;
}

static bool same_data(const Tensor &a, const Tensor &b, float tol = 0.0f) {
  if (a.shape() != b.shape())
    return false;
  for (std::size_t n = 0; n < a.numel(); ++n)
    if (std::abs(a.data()[n] - b.data()[n]) > tol)
      return false;
  return true;
}

// =====================================================================
// gemm
// =====================================================================

// A (2x3) lives inside a 4x5 buffer (lda = 5); the rest is garbage.
static void test_gemm_submatrix_input() {
  std::vector<float> bufA(20, -7.0f);
  const float a_vals[2][3] = {{1, 2, 3}, {4, 5, 6}};
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      bufA[i * 5 + j] = a_vals[i][j];

  std::vector<float> B = {7, 8, 9, 10, 11, 12}; // 3x2 packed
  std::vector<float> C(4, 0.0f);                // 2x2 packed

  gemm(bufA.data(), B.data(), C.data(), 2, 3, 2, /*lda=*/5, 1, /*ldb=*/2, 1,
       /*ldc=*/2, 1, false, false, false);
  EXPECT(C[0] == 58 && C[1] == 64 && C[2] == 139 && C[3] == 154);
  std::cout << "gemm (sub-matrix input, lda > cols): ok\n";
}

// C (2x2) lives inside a 2x4 buffer (ldc = 4). Padding must stay untouched.
static void test_gemm_padded_output() {
  std::vector<float> A = {1, 2, 3, 4, 5, 6};    // 2x3
  std::vector<float> B = {7, 8, 9, 10, 11, 12}; // 3x2
  std::vector<float> C(8, 99.0f);               // 2 rows, ldc = 4

  gemm(A.data(), B.data(), C.data(), 2, 3, 2, 3, 1, 2, 1, /*ldc=*/4, 1, false,
       false, false);
  EXPECT(C[0] == 58 && C[1] == 64 && C[4] == 139 && C[5] == 154);
  EXPECT(C[2] == 99 && C[3] == 99 && C[6] == 99 && C[7] == 99);
  std::cout << "gemm (padded output, ldc > N): ok\n";
}

// Reusing C with accumulate = false must overwrite ALL rows, not double up.
static void test_gemm_reuse_overwrites() {
  std::vector<float> A = {1, 2, 3, 4, 5, 6};
  std::vector<float> B = {7, 8, 9, 10, 11, 12};
  std::vector<float> C(4, 0.0f);

  gemm(A.data(), B.data(), C.data(), 2, 3, 2, 3, 1, 2, 1, 2, 1, false, false,
       false);
  gemm(A.data(), B.data(), C.data(), 2, 3, 2, 3, 1, 2, 1, 2, 1, false, false,
       false);
  EXPECT(C[0] == 58 && C[1] == 64 && C[2] == 139 && C[3] == 154);

  gemm(A.data(), B.data(), C.data(), 2, 3, 2, 3, 1, 2, 1, 2, 1, false, false,
       true);
  EXPECT(C[0] == 116 && C[3] == 308); // accumulate doubles it
  std::cout << "gemm (overwrite vs accumulate on reuse): ok\n";
}

// =====================================================================
// Linear: structure and initialization
// =====================================================================

static void test_linear_structure() {
  Linear with_bias(4, 5);
  Linear no_bias(4, 5, /*bias=*/false);
  EXPECT(with_bias.parameters().size() == 2);
  EXPECT(with_bias.gradients().size() == 2);
  EXPECT(no_bias.parameters().size() == 1);
  EXPECT(no_bias.gradients().size() == 1);

  // Parameter / gradient pairs must have matching shapes, in matching order.
  auto p = with_bias.parameters();
  auto g = with_bias.gradients();
  for (std::size_t i = 0; i < p.size(); ++i)
    EXPECT(p[i]->shape() == g[i]->shape());

  // PyTorch layout: W is (out, in), bias is (out).
  EXPECT(p[0]->shape() == std::vector<std::size_t>({5, 4}));
  EXPECT(p[1]->shape() == std::vector<std::size_t>({5}));
  std::cout << "linear (structure, shapes, bias flag): ok\n";
}

static void test_linear_init() {
  Linear a(4, 5, true, 7), b(4, 5, true, 7), c(4, 5, true, 8);
  EXPECT(same_data(*a.parameters()[0], *b.parameters()[0]));  // same seed
  EXPECT(!same_data(*a.parameters()[0], *c.parameters()[0])); // different seed

  const float bound = std::sqrt(6.0f / 9.0f);
  const Tensor &W = *a.parameters()[0];
  for (std::size_t n = 0; n < W.numel(); ++n)
    EXPECT(std::abs(W.data()[n]) <= bound);

  const Tensor &bias = *a.parameters()[1];
  for (std::size_t n = 0; n < bias.numel(); ++n)
    EXPECT(bias.data()[n] == 0.0f);
  std::cout << "linear (seeded init, Xavier bound, zero bias): ok\n";
}

// =====================================================================
// Linear: forward
// =====================================================================

// in = 3, out = 2.  W = [[1,2,3],[4,5,6]], b = [10,20].
static void make_hand_layer(Linear &layer) {
  auto p = layer.parameters();
  set_values(*p[0], {1, 2, 3, 4, 5, 6});
  if (p.size() > 1)
    set_values(*p[1], {10, 20});
}

static void test_linear_forward_hand() {
  Linear layer(3, 2);
  make_hand_layer(layer);

  Tensor X({2, 3}), Y({2, 2});
  set_values(X, {1, 0, -1, 2, 1, 0});
  Y.fill(99.0f); // stale values must not leak through

  layer.forward(X, Y);
  EXPECT(Y.data()[0] == 8 && Y.data()[1] == 18);
  EXPECT(Y.data()[2] == 14 && Y.data()[3] == 33);
  std::cout << "linear forward (hand-checked, stale output): ok\n";
}

static void test_linear_forward_no_bias() {
  Linear layer(3, 2, /*bias=*/false);
  make_hand_layer(layer);

  Tensor X({2, 3}), Y({2, 2});
  set_values(X, {1, 0, -1, 2, 1, 0});
  Y.fill(99.0f);

  layer.forward(X, Y);
  EXPECT(Y.data()[0] == -2 && Y.data()[1] == -2);
  EXPECT(Y.data()[2] == 4 && Y.data()[3] == 13);
  std::cout << "linear forward (no bias, stale output): ok\n";
}

// With W = 0, every output row must equal the bias (tests per-column bias).
static void test_linear_forward_bias_only() {
  Linear layer(3, 2);
  auto p = layer.parameters();
  p[0]->fill(0.0f);
  set_values(*p[1], {10, 20});

  std::mt19937 rng(1);
  Tensor X({5, 3}), Y({5, 2});
  fill_random(X, rng);
  Y.fill(99.0f);

  layer.forward(X, Y);
  for (std::size_t r = 0; r < 5; ++r) {
    EXPECT(Y.data()[r * 2] == 10.0f);
    EXPECT(Y.data()[r * 2 + 1] == 20.0f);
  }
  std::cout << "linear forward (bias broadcast over rows): ok\n";
}

// A rank-3 input must behave exactly like the same data flattened to rank 2.
static void test_linear_forward_rank3() {
  std::mt19937 rng(2);
  Linear layer(4, 5, true, 3);
  fill_random(*layer.parameters()[1], rng); // non-zero bias

  Tensor X3({2, 3, 4}), X2({6, 4});
  fill_random(X3, rng);
  for (std::size_t n = 0; n < X3.numel(); ++n)
    X2.data()[n] = X3.data()[n];

  Tensor Y3({2, 3, 5}), Y2({6, 5});
  layer.forward(X3, Y3);
  layer.forward(X2, Y2);
  for (std::size_t n = 0; n < Y3.numel(); ++n)
    EXPECT(Y3.data()[n] == Y2.data()[n]);
  std::cout << "linear forward (rank 3 == flattened rank 2): ok\n";
}

// =====================================================================
// Linear: backward
// =====================================================================

// Same hand example, dY = all ones:
//   dX rows = column sums of W = [5,7,9]
//   dW rows = column sums of X = [3,1,-1]
//   db      = [2,2]
static void test_linear_backward_hand() {
  Linear layer(3, 2);
  make_hand_layer(layer);

  Tensor X({2, 3}), dY({2, 2}), dX({2, 3});
  set_values(X, {1, 0, -1, 2, 1, 0});
  dY.fill(1.0f);

  layer.zero_grad();
  layer.backward(X, dY, dX, /*accumulate=*/false);

  for (std::size_t r = 0; r < 2; ++r) {
    EXPECT(dX.data()[r * 3 + 0] == 5);
    EXPECT(dX.data()[r * 3 + 1] == 7);
    EXPECT(dX.data()[r * 3 + 2] == 9);
  }
  const Tensor &dW = *layer.gradients()[0];
  const Tensor &db = *layer.gradients()[1];
  for (std::size_t r = 0; r < 2; ++r) {
    EXPECT(dW.data()[r * 3 + 0] == 3);
    EXPECT(dW.data()[r * 3 + 1] == 1);
    EXPECT(dW.data()[r * 3 + 2] == -1);
  }
  EXPECT(db.data()[0] == 2 && db.data()[1] == 2);
  std::cout << "linear backward (hand-checked): ok\n";
}

static void test_linear_backward_accumulation() {
  Linear layer(3, 2);
  make_hand_layer(layer);
  Tensor X({2, 3}), dY({2, 2}), dX({2, 3});
  set_values(X, {1, 0, -1, 2, 1, 0});
  dY.fill(1.0f);

  // dW and db always accumulate; zero_grad() (via the base) clears them.
  layer.zero_grad();
  layer.backward(X, dY, dX, false);
  layer.backward(X, dY, dX, false);
  EXPECT(layer.gradients()[0]->data()[0] == 6); // 2 x 3
  EXPECT(layer.gradients()[1]->data()[0] == 4); // 2 x 2

  Module &base = layer;
  base.zero_grad();
  for (Tensor *g : base.gradients())
    for (std::size_t n = 0; n < g->numel(); ++n)
      EXPECT(g->data()[n] == 0.0f);

  // dX: overwrite vs accumulate.
  dX.fill(10.0f);
  layer.backward(X, dY, dX, /*accumulate=*/true);
  EXPECT(dX.data()[0] == 15 && dX.data()[1] == 17 && dX.data()[2] == 19);
  layer.backward(X, dY, dX, /*accumulate=*/false);
  EXPECT(dX.data()[0] == 5 && dX.data()[1] == 7 && dX.data()[2] == 9);
  std::cout << "linear backward (accumulation, zero_grad via Module&): ok\n";
}

// Gradcheck helper: L = sum(Y * R), so dL/dY == R.
static void run_linear_gradcheck(bool bias, std::vector<std::size_t> x_shape,
                                 std::size_t out, const char *name) {
  std::mt19937 rng(100);
  const std::size_t in = x_shape.back();
  std::vector<std::size_t> y_shape = x_shape;
  y_shape.back() = out;

  Linear layer(in, out, bias, /*seed=*/5);
  Tensor X(x_shape), Y(y_shape), R(y_shape), dX(x_shape);
  fill_random(X, rng);
  fill_random(R, rng);

  auto params = layer.parameters();
  auto grads = layer.gradients();
  if (bias)
    fill_random(*params[1], rng); // non-zero bias

  auto loss = [&]() {
    layer.forward(X, Y);
    return dot_loss(Y, R);
  };

  layer.zero_grad();
  layer.backward(X, /*dY=*/R, dX, /*accumulate=*/false);

  // The loss is linear in each input, so a large eps is safe and cuts
  // float rounding noise.
  const double eps = 1e-2, tol = 2e-2;

  GradcheckResult rx = check_gradient(X, dX, loss, eps, tol);
  GradcheckResult rw = check_gradient(*params[0], *grads[0], loss, eps, tol);
  std::cout << name << ": max rel err  X " << rx.max_rel_error << "  W "
            << rw.max_rel_error;
  EXPECT(rx.passed);
  EXPECT(rw.passed);
  if (bias) {
    GradcheckResult rb = check_gradient(*params[1], *grads[1], loss, eps, tol);
    std::cout << "  b " << rb.max_rel_error;
    EXPECT(rb.passed);
  }
  std::cout << "\n";

  // Self-test: a wrong dW must FAIL the checker.
  Tensor dW_bad = grads[0]->clone();
  for (std::size_t n = 0; n < dW_bad.numel(); ++n)
    dW_bad.data()[n] *= 1.1f;
  EXPECT(!check_gradient(*params[0], dW_bad, loss, eps, tol).passed);
}

static void test_linear_gradcheck() {
  run_linear_gradcheck(true, {4, 3}, 5, "linear gradcheck (bias, rank 2)");
  run_linear_gradcheck(false, {4, 3}, 5, "linear gradcheck (no bias, rank 2)");
  run_linear_gradcheck(true, {2, 3, 4}, 5, "linear gradcheck (bias, rank 3)");
  std::cout << "linear gradcheck (incl. self-tests): ok\n";
}

void tests::run_linear() {
  test_gemm_submatrix_input();
  test_gemm_padded_output();
  test_gemm_reuse_overwrites();

  test_linear_structure();
  test_linear_init();

  test_linear_forward_hand();
  test_linear_forward_no_bias();
  test_linear_forward_bias_only();
  test_linear_forward_rank3();

  test_linear_backward_hand();
  test_linear_backward_accumulation();
  test_linear_gradcheck();

  std::cout << "all linear tests passed\n";
}
