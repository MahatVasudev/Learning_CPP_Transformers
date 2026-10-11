#include "functional/activations.h"
#include "functional/gradcheck.h"
#include "functional/tensor.h"
#include "tests/expect.h"
#include "tests/tests.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

using namespace tests;
using namespace functional;

static void fill_random(Tensor &t, std::mt19937 &rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t n = 0; n < t.numel(); ++n)
    t.data()[n] = dist(rng);
}

// Magnitudes in [0.1, 1] with random sign: keeps x +/- eps away from ReLU's
// kink at 0, so the numeric gradient is valid.
static void fill_away_from_zero(Tensor &t, std::mt19937 &rng) {
  std::uniform_real_distribution<float> mag(0.1f, 1.0f);
  std::bernoulli_distribution neg(0.5);
  for (std::size_t n = 0; n < t.numel(); ++n)
    t.data()[n] = neg(rng) ? -mag(rng) : mag(rng);
}

static double dot_loss(const Tensor &Y, const Tensor &R) {
  double L = 0.0;
  for (std::size_t n = 0; n < Y.numel(); ++n)
    L += static_cast<double>(Y.data()[n]) * R.data()[n];
  return L;
}

static Tensor from_values(std::vector<std::size_t> shape,
                          const std::vector<float> &vals) {
  Tensor t(std::move(shape));
  EXPECT(t.numel() == vals.size());
  for (std::size_t n = 0; n < vals.size(); ++n)
    t.data()[n] = vals[n];
  return t;
}

// ---------- ReLU ----------

static void test_relu_hand() {
  ReLU relu;
  Tensor X = from_values({1, 4}, {-2.0f, -0.5f, 0.5f, 3.0f});
  Tensor Y({1, 4}), dY({1, 4}), dX({1, 4});
  dY.fill(1.0f);

  relu.forward(X, Y);
  EXPECT(Y.data()[0] == 0.0f && Y.data()[1] == 0.0f);
  EXPECT(Y.data()[2] == 0.5f && Y.data()[3] == 3.0f);

  relu.backward(X, Y, dY, dX, /*accumulate=*/false);
  EXPECT(dX.data()[0] == 0.0f && dX.data()[1] == 0.0f);
  EXPECT(dX.data()[2] == 1.0f && dX.data()[3] == 1.0f);
  std::cout << "relu (hand-checked): ok\n";
}

static void test_relu_accumulate() {
  ReLU relu;
  Tensor X = from_values({1, 4}, {-2.0f, -0.5f, 0.5f, 3.0f});
  Tensor Y({1, 4}), dY({1, 4}), dX({1, 4});
  dY.fill(1.0f);
  relu.forward(X, Y);

  dX.fill(10.0f);
  relu.backward(X, Y, dY, dX, /*accumulate=*/true);
  EXPECT(dX.data()[0] == 10.0f && dX.data()[1] == 10.0f);
  EXPECT(dX.data()[2] == 11.0f && dX.data()[3] == 11.0f);

  dX.fill(10.0f);
  relu.backward(X, Y, dY, dX, /*accumulate=*/false); // must overwrite
  EXPECT(dX.data()[0] == 0.0f && dX.data()[2] == 1.0f);
  std::cout << "relu (accumulate vs overwrite): ok\n";
}

static void test_relu_gradcheck() {
  std::mt19937 rng(10);
  ReLU relu;
  Tensor X({3, 4, 5}), Y({3, 4, 5}), R({3, 4, 5}), dX({3, 4, 5});
  fill_away_from_zero(X, rng);
  fill_random(R, rng);

  auto loss = [&]() {
    relu.forward(X, Y);
    return dot_loss(Y, R);
  };

  loss(); // populate Y for backward
  relu.backward(X, Y, /*dY=*/R, dX, /*accumulate=*/false);

  GradcheckResult r = check_gradient(X, dX, loss);
  std::cout << "relu gradcheck: max rel err " << r.max_rel_error << "\n";
  EXPECT(r.passed);

  // Self-test: a gradient with the mask flipped must FAIL.
  Tensor dX_bad({3, 4, 5});
  for (std::size_t n = 0; n < dX_bad.numel(); ++n)
    dX_bad.data()[n] = X.data()[n] < 0 ? R.data()[n] : 0.0f;
  EXPECT(!check_gradient(X, dX_bad, loss).passed);
  std::cout << "relu gradcheck (incl. self-test): ok\n";
}

// ---------- Softmax ----------

static void test_softmax_hand() {
  Softmax sm;
  Tensor X({1, 2}), Y({1, 2});
  X.fill(0.0f);
  sm.forward(X, Y);
  EXPECT(std::abs(Y.data()[0] - 0.5f) < 1e-6f);
  EXPECT(std::abs(Y.data()[1] - 0.5f) < 1e-6f);
  std::cout << "softmax (hand-checked): ok\n";
}

static void test_softmax_rows_sum_to_one() {
  std::mt19937 rng(11);
  Softmax sm;
  Tensor X({2, 3, 5}), Y({2, 3, 5});
  fill_random(X, rng);
  sm.forward(X, Y);

  const std::size_t cols = 5, rows = 6;
  for (std::size_t r = 0; r < rows; ++r) {
    float sum = 0.0f;
    for (std::size_t c = 0; c < cols; ++c) {
      float v = Y.data()[r * cols + c];
      EXPECT(v > 0.0f && v < 1.0f);
      sum += v;
    }
    EXPECT(std::abs(sum - 1.0f) < 1e-5f);
  }
  std::cout << "softmax (rows sum to 1, rank 3): ok\n";
}

static void test_softmax_stability() {
  Softmax sm;
  Tensor big = from_values({1, 2}, {1000.0f, 1001.0f});
  Tensor small = from_values({1, 2}, {0.0f, 1.0f});
  Tensor Yb({1, 2}), Ys({1, 2});
  sm.forward(big, Yb);
  sm.forward(small, Ys);
  for (int i = 0; i < 2; ++i) {
    EXPECT(std::isfinite(Yb.data()[i]));
    EXPECT(std::abs(Yb.data()[i] - Ys.data()[i]) < 1e-6f);
  }

  Tensor neg = from_values({1, 3}, {-1000.0f, -1000.0f, -1000.0f});
  Tensor Yn({1, 3});
  sm.forward(neg, Yn);
  for (int i = 0; i < 3; ++i)
    EXPECT(std::abs(Yn.data()[i] - 1.0f / 3.0f) < 1e-6f);
  std::cout << "softmax (numerical stability): ok\n";
}

static void test_softmax_accumulate() {
  std::mt19937 rng(12);
  Softmax sm;
  Tensor X({2, 4}), Y({2, 4}), dY({2, 4}), once({2, 4}), twice({2, 4});
  fill_random(X, rng);
  fill_random(dY, rng);
  sm.forward(X, Y);

  sm.backward(X, Y, dY, once, false);
  sm.backward(X, Y, dY, twice, false);
  sm.backward(X, Y, dY, twice, true); // now holds 2x
  for (std::size_t n = 0; n < once.numel(); ++n)
    EXPECT(std::abs(twice.data()[n] - 2.0f * once.data()[n]) < 1e-6f);
  std::cout << "softmax (accumulate): ok\n";
}

static void test_softmax_gradcheck() {
  std::mt19937 rng(13);
  Softmax sm;
  Tensor X({2, 3, 5}), Y({2, 3, 5}), R({2, 3, 5}), dX({2, 3, 5});
  fill_random(X, rng);
  fill_random(R, rng);

  auto loss = [&]() {
    sm.forward(X, Y);
    return dot_loss(Y, R);
  };

  loss();
  sm.backward(X, Y, /*dY=*/R, dX, false);

  // Looser eps/tol than matmul: softmax outputs are float, and some
  // gradient entries are small, which makes relative error noisy.
  const double eps = 1e-2, tol = 5e-2;
  GradcheckResult r = check_gradient(X, dX, loss, eps, tol);
  std::cout << "softmax gradcheck: max rel err " << r.max_rel_error << "\n";
  EXPECT(r.passed);

  // Property: each row of dX sums to ~0 (softmax is invariant to adding a
  // constant to a whole row of logits).
  const std::size_t cols = 5, rows = 6;
  for (std::size_t row = 0; row < rows; ++row) {
    double s = 0.0;
    for (std::size_t c = 0; c < cols; ++c)
      s += dX.data()[row * cols + c];
    EXPECT(std::abs(s) < 1e-5);
  }

  // Self-test: dropping the "- dot" term must FAIL the checker.
  Tensor dX_bad({2, 3, 5});
  for (std::size_t n = 0; n < dX_bad.numel(); ++n)
    dX_bad.data()[n] = Y.data()[n] * R.data()[n];
  EXPECT(!check_gradient(X, dX_bad, loss, eps, tol).passed);
  std::cout << "softmax gradcheck (incl. property + self-test): ok\n";
}

// ---------- polymorphism ----------

static void test_polymorphic_dispatch() {
  std::vector<std::unique_ptr<Activation>> acts;
  acts.push_back(std::make_unique<ReLU>());
  acts.push_back(std::make_unique<Softmax>());

  Tensor X = from_values({1, 3}, {-1.0f, 0.5f, 2.0f});
  Tensor Y({1, 3}), dY({1, 3}), dX({1, 3});
  dY.fill(1.0f);

  // ReLU through the base interface.
  acts[0]->forward(X, Y);
  EXPECT(Y.data()[0] == 0.0f && Y.data()[1] == 0.5f && Y.data()[2] == 2.0f);
  acts[0]->backward(X, Y, dY, dX, false);
  EXPECT(dX.data()[0] == 0.0f && dX.data()[2] == 1.0f);

  // Softmax through the base interface.
  acts[1]->forward(X, Y);
  float sum = Y.data()[0] + Y.data()[1] + Y.data()[2];
  EXPECT(std::abs(sum - 1.0f) < 1e-5f);
  // acts destroyed here: sanitizers would flag a missing virtual destructor.
  std::cout << "polymorphic dispatch via unique_ptr<Activation>: ok\n";
}

void tests::run_activations() {
  test_relu_hand();
  test_relu_accumulate();
  test_relu_gradcheck();

  test_softmax_hand();
  test_softmax_rows_sum_to_one();
  test_softmax_stability();
  test_softmax_accumulate();
  test_softmax_gradcheck();

  test_polymorphic_dispatch();
  std::cout << "all activation tests passed\n";
}
