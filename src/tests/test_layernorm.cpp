#include "functional/gradcheck.h"
#include "functional/module.h" // adjust if LayerNorm lives elsewhere
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

static const double EPS = 1e-5; // LayerNorm's default epsilon

// One place to adapt if backward's signature changes (e.g. Y dropped).
static void ln_backward(LayerNorm &ln, const Tensor &X, const Tensor &Y,
                        const Tensor &dY, Tensor &dX, bool accumulate) {
  ln.backward(X, dY, dX, accumulate);
}

// ---------- helpers ----------

static void fill_random(Tensor &t, std::mt19937 &rng, float lo = -1.0f,
                        float hi = 1.0f) {
  std::uniform_real_distribution<float> dist(lo, hi);
  for (std::size_t n = 0; n < t.numel(); ++n)
    t.data()[n] = dist(rng);
}

static void set_values(Tensor &t, const std::vector<float> &vals) {
  EXPECT(t.numel() == vals.size());
  for (std::size_t n = 0; n < vals.size(); ++n)
    t.data()[n] = vals[n];
}

static double dot_loss(const Tensor &Y, const Tensor &R) {
  double L = 0.0;
  for (std::size_t n = 0; n < Y.numel(); ++n)
    L += static_cast<double>(Y.data()[n]) * R.data()[n];
  return L;
}

// Reference statistics for one row, computed in double.
static void row_stats(const float *x, std::size_t D, double &mean, double &rstd,
                      double eps = EPS) {
  mean = 0.0;
  for (std::size_t j = 0; j < D; ++j)
    mean += x[j];
  mean /= static_cast<double>(D);
  double var = 0.0;
  for (std::size_t j = 0; j < D; ++j) {
    const double c = x[j] - mean;
    var += c * c;
  }
  var /= static_cast<double>(D);
  rstd = 1.0 / std::sqrt(var + eps);
}

static bool all_finite(const Tensor &t) {
  for (std::size_t n = 0; n < t.numel(); ++n)
    if (!std::isfinite(t.data()[n]))
      return false;
  return true;
}

static void report(const char *name, const GradcheckResult &r) {
  std::cout << "  " << name << ": max rel err " << r.max_rel_error;
  if (!r.passed)
    std::cout << "  [worst idx " << r.worst_index << ", analytic " << r.analytic
              << ", numeric " << r.numeric << "]";
  std::cout << "\n";
}

// =====================================================================
// structure
// =====================================================================

static void test_structure() {
  LayerNorm ln(6);
  Module &base = ln;
  EXPECT(base.parameters().size() == 2);
  EXPECT(base.gradients().size() == 2);

  const std::vector<std::size_t> shape6 = {6};
  for (std::size_t i = 0; i < 2; ++i) {
    EXPECT(base.parameters()[i]->shape() == shape6);
    EXPECT(base.gradients()[i]->shape() == shape6);
  }
  for (std::size_t j = 0; j < 6; ++j) {
    EXPECT(base.parameters()[0]->data()[j] == 1.0f); // gamma
    EXPECT(base.parameters()[1]->data()[j] == 0.0f); // beta
    EXPECT(base.gradients()[0]->data()[j] == 0.0f);
    EXPECT(base.gradients()[1]->data()[j] == 0.0f);
  }
  std::cout << "layernorm (structure, init gamma=1 beta=0): ok\n";
}

// =====================================================================
// forward
// =====================================================================

// Row [1,2,3,4]: mean 2.5, variance 1.25.
static void test_forward_hand() {
  LayerNorm ln(4);
  Tensor X({1, 4}), Y({1, 4});
  set_values(X, {1, 2, 3, 4});
  ln.forward(X, Y);

  const double rstd = 1.0 / std::sqrt(1.25 + EPS);
  for (std::size_t j = 0; j < 4; ++j)
    EXPECT(std::abs(Y.data()[j] - (j - 1.5) * rstd) < 1e-5);

  // Literal values, as a sanity check on the reference above.
  EXPECT(std::abs(Y.data()[0] + 1.3416f) < 1e-3f);
  EXPECT(std::abs(Y.data()[1] + 0.4472f) < 1e-3f);
  EXPECT(std::abs(Y.data()[2] - 0.4472f) < 1e-3f);
  EXPECT(std::abs(Y.data()[3] - 1.3416f) < 1e-3f);
  std::cout << "layernorm forward (hand-checked): ok\n";
}

// A custom epsilon must be used: eps = 1 gives y = (x - 2.5) / 1.5.
static void test_forward_custom_epsilon() {
  LayerNorm ln(4, 1.0f);
  Tensor X({1, 4}), Y({1, 4});
  set_values(X, {1, 2, 3, 4});
  ln.forward(X, Y);
  EXPECT(std::abs(Y.data()[0] + 1.0f) < 1e-5f);
  EXPECT(std::abs(Y.data()[1] + 1.0f / 3.0f) < 1e-5f);
  EXPECT(std::abs(Y.data()[2] - 1.0f / 3.0f) < 1e-5f);
  EXPECT(std::abs(Y.data()[3] - 1.0f) < 1e-5f);
  std::cout << "layernorm forward (custom epsilon): ok\n";
}

// gamma and beta must be applied per COLUMN.
static void test_forward_gamma_beta() {
  LayerNorm ln(4);
  set_values(*ln.parameters()[0], {1, 2, 3, 4});
  set_values(*ln.parameters()[1], {10, 20, 30, 40});

  std::mt19937 rng(1);
  Tensor X({3, 4}), Y({3, 4});
  fill_random(X, rng);
  ln.forward(X, Y);

  for (std::size_t r = 0; r < 3; ++r) {
    double mean, rstd;
    row_stats(X.data() + r * 4, 4, mean, rstd);
    for (std::size_t j = 0; j < 4; ++j) {
      const double xhat = (X.data()[r * 4 + j] - mean) * rstd;
      const double expected = (j + 1) * xhat + 10.0 * (j + 1);
      EXPECT(std::abs(Y.data()[r * 4 + j] - expected) < 1e-4);
    }
  }
  std::cout << "layernorm forward (gamma/beta per column): ok\n";
}

// With gamma=1, beta=0: every row has mean ~0 and variance ~1, even when the
// input has a large mean and spread.
static void test_forward_statistics() {
  std::mt19937 rng(2);
  LayerNorm ln(16);
  Tensor X({3, 5, 16}), Y({3, 5, 16});
  fill_random(X, rng, 2.0f, 8.0f);
  ln.forward(X, Y);

  for (std::size_t r = 0; r < 15; ++r) {
    double mean = 0.0, var = 0.0;
    for (std::size_t j = 0; j < 16; ++j)
      mean += Y.data()[r * 16 + j];
    mean /= 16.0;
    for (std::size_t j = 0; j < 16; ++j) {
      const double c = Y.data()[r * 16 + j] - mean;
      var += c * c;
    }
    var /= 16.0;
    EXPECT(std::abs(mean) < 1e-4);
    EXPECT(std::abs(var - 1.0) < 1e-3);
  }
  std::cout << "layernorm forward (rows: mean 0, var 1): ok\n";
}

// Constant row: variance 0, so eps must keep everything finite; y == beta.
static void test_forward_constant_row() {
  std::mt19937 rng(3);
  LayerNorm ln(6);
  fill_random(*ln.parameters()[1], rng); // random beta

  Tensor X({2, 6}), Y({2, 6});
  X.fill(3.0f);
  ln.forward(X, Y);
  EXPECT(all_finite(Y));
  for (std::size_t r = 0; r < 2; ++r)
    for (std::size_t j = 0; j < 6; ++j)
      EXPECT(std::abs(Y.data()[r * 6 + j] - ln.beta().data()[j]) < 1e-6f);

  // Backward on a constant row must also stay finite.
  Tensor dY({2, 6}), dX({2, 6});
  fill_random(dY, rng);
  ln_backward(ln, X, Y, dY, dX, false);
  EXPECT(all_finite(dX));
  std::cout << "layernorm (constant row stays finite, y == beta): ok\n";
}

static void test_forward_stale_output() {
  std::mt19937 rng(4);
  LayerNorm ln(8);
  fill_random(*ln.parameters()[0], rng);
  fill_random(*ln.parameters()[1], rng);
  Tensor X({4, 8}), fresh({4, 8}), stale({4, 8});
  fill_random(X, rng);
  stale.fill(99.0f);

  ln.forward(X, fresh);
  ln.forward(X, stale);
  for (std::size_t n = 0; n < fresh.numel(); ++n)
    EXPECT(fresh.data()[n] == stale.data()[n]);
  std::cout << "layernorm forward (stale output overwritten): ok\n";
}

static void test_forward_rank3() {
  std::mt19937 rng(5);
  LayerNorm ln(8);
  fill_random(*ln.parameters()[0], rng);
  fill_random(*ln.parameters()[1], rng);

  Tensor X3({2, 3, 8}), X2({6, 8}), Y3({2, 3, 8}), Y2({6, 8});
  fill_random(X3, rng);
  for (std::size_t n = 0; n < X3.numel(); ++n)
    X2.data()[n] = X3.data()[n];

  ln.forward(X3, Y3);
  ln.forward(X2, Y2);
  for (std::size_t n = 0; n < Y3.numel(); ++n)
    EXPECT(Y3.data()[n] == Y2.data()[n]);
  std::cout << "layernorm forward (rank 3 == flattened rank 2): ok\n";
}

// =====================================================================
// backward
// =====================================================================

// Shared setup for the backward tests.
struct Setup {
  std::size_t D;
  LayerNorm ln;
  Tensor X, Y, R, dX;

  Setup(std::vector<std::size_t> shape, unsigned seed)
      : D(shape.back()), ln(shape.back()), X(shape), Y(shape), R(shape),
        dX(shape) {
    std::mt19937 rng(seed);
    fill_random(X, rng);
    fill_random(R, rng);
    fill_random(*ln.parameters()[0], rng, 0.5f, 1.5f); // random gamma
    fill_random(*ln.parameters()[1], rng);             // random beta
  }

  double loss() {
    ln.forward(X, Y);
    return dot_loss(Y, R);
  }
};

static void test_backward_gradcheck() {
  Setup s({3, 4, 8}, 10);
  auto params = s.ln.parameters();
  auto grads = s.ln.gradients();

  auto loss = [&]() { return s.loss(); };

  loss(); // populate Y
  s.ln.zero_grad();
  ln_backward(s.ln, s.X, s.Y, /*dY=*/s.R, s.dX, false);

  // Looser eps/tol than matmul: layernorm is nonlinear in X, and the
  // outputs are float.
  const double eps = 1e-2, tol = 5e-2;

  std::cout << "layernorm gradcheck (rank 3):\n";
  GradcheckResult rx = check_gradient(s.X, s.dX, loss, eps, tol);
  GradcheckResult rg = check_gradient(*params[0], *grads[0], loss, eps, tol);
  GradcheckResult rb = check_gradient(*params[1], *grads[1], loss, eps, tol);
  report("dX    ", rx);
  report("dgamma", rg);
  report("dbeta ", rb);
  EXPECT(rx.passed);
  EXPECT(rg.passed);
  EXPECT(rb.passed);

  // Self-test 1: dX without the xhat * mean(dxhat * xhat) term must FAIL.
  Tensor dX_bad(s.X.shape());
  const std::size_t rows = s.X.numel() / s.D;
  for (std::size_t r = 0; r < rows; ++r) {
    double mean, rstd;
    row_stats(s.X.data() + r * s.D, s.D, mean, rstd);
    double m1 = 0.0;
    for (std::size_t j = 0; j < s.D; ++j)
      m1 += static_cast<double>(s.R.data()[r * s.D + j]) * params[0]->data()[j];
    m1 /= static_cast<double>(s.D);
    for (std::size_t j = 0; j < s.D; ++j) {
      const double dxhat =
          static_cast<double>(s.R.data()[r * s.D + j]) * params[0]->data()[j];
      dX_bad.data()[r * s.D + j] = static_cast<float>(rstd * (dxhat - m1));
    }
  }
  EXPECT(!check_gradient(s.X, dX_bad, loss, eps, tol).passed);

  // Self-test 2: a wrong dgamma must FAIL.
  Tensor dg_bad = grads[0]->clone();
  for (std::size_t n = 0; n < dg_bad.numel(); ++n)
    dg_bad.data()[n] *= 1.1f;
  EXPECT(!check_gradient(*params[0], dg_bad, loss, eps, tol).passed);

  std::cout << "layernorm gradcheck (incl. self-tests): ok\n";
}

// Independent algebraic checks on dX, valid for ANY gamma:
//   * each row of dX sums to ~0   (shifting a row doesn't change the output)
//   * each row of dX is ~orthogonal to xhat (scaling a row doesn't either)
static void test_backward_invariants() {
  Setup s({4, 6, 8}, 11);
  s.loss();
  s.ln.zero_grad();
  ln_backward(s.ln, s.X, s.Y, s.R, s.dX, false);

  const std::size_t rows = s.X.numel() / s.D;
  for (std::size_t r = 0; r < rows; ++r) {
    double mean, rstd;
    row_stats(s.X.data() + r * s.D, s.D, mean, rstd);
    double sum = 0.0, dot = 0.0, m2 = 0.0;
    for (std::size_t j = 0; j < s.D; ++j) {
      const double xhat = (s.X.data()[r * s.D + j] - mean) * rstd;
      const double dxhat =
          static_cast<double>(s.R.data()[r * s.D + j]) * s.ln.gamma().data()[j];
      const double g = s.dX.data()[r * s.D + j];
      sum += g;
      dot += g * xhat;
      m2 += dxhat * xhat;
    }
    m2 /= static_cast<double>(s.D);

    // The residual from the eps in the denominator, predicted exactly:
    const double predicted = rstd * rstd * rstd * EPS * s.D * m2;
    EXPECT(std::abs(sum) < 1e-4);
    EXPECT(std::abs(dot - predicted) < 1e-4);
  }
  std::cout << "layernorm backward (row-sum and xhat-orthogonality): ok\n";
}

static void test_backward_accumulation() {
  Setup s({3, 4}, 12);
  s.loss();

  // dgamma / dbeta always accumulate; zero_grad() (via Module&) clears them.
  s.ln.zero_grad();
  ln_backward(s.ln, s.X, s.Y, s.R, s.dX, false);
  Tensor dg1 = s.ln.gradients()[0]->clone();
  Tensor db1 = s.ln.gradients()[1]->clone();
  ln_backward(s.ln, s.X, s.Y, s.R, s.dX, false);
  for (std::size_t j = 0; j < s.D; ++j) {
    EXPECT(std::abs(s.ln.gradients()[0]->data()[j] - 2.0f * dg1.data()[j]) <
           1e-5f * (1.0f + std::abs(dg1.data()[j])));
    EXPECT(std::abs(s.ln.gradients()[1]->data()[j] - 2.0f * db1.data()[j]) <
           1e-5f * (1.0f + std::abs(db1.data()[j])));
  }

  Module &base = s.ln;
  base.zero_grad();
  for (Tensor *g : base.gradients())
    for (std::size_t n = 0; n < g->numel(); ++n)
      EXPECT(g->data()[n] == 0.0f);

  // dX: overwrite vs accumulate.
  Tensor ref(s.X.shape());
  ln_backward(s.ln, s.X, s.Y, s.R, ref, false);
  s.dX.fill(10.0f);
  ln_backward(s.ln, s.X, s.Y, s.R, s.dX, true);
  for (std::size_t n = 0; n < ref.numel(); ++n)
    EXPECT(std::abs(s.dX.data()[n] - (ref.data()[n] + 10.0f)) < 1e-5f);
  ln_backward(s.ln, s.X, s.Y, s.R, s.dX, false);
  for (std::size_t n = 0; n < ref.numel(); ++n)
    EXPECT(std::abs(s.dX.data()[n] - ref.data()[n]) < 1e-6f);
  std::cout << "layernorm backward (accumulation, zero_grad via Module&): ok\n";
}

void tests::run_layernorm() {
  test_structure();

  test_forward_hand();
  test_forward_custom_epsilon();
  test_forward_gamma_beta();
  test_forward_statistics();
  test_forward_constant_row();
  test_forward_stale_output();
  test_forward_rank3();

  test_backward_gradcheck();
  test_backward_invariants();
  test_backward_accumulation();

  std::cout << "all layernorm tests passed\n";
}
