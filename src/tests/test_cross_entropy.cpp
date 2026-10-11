#include "functional/activations.h"
#include "functional/cross_entropy.h"
#include "functional/gradcheck.h"
#include "functional/tensor.h"
#include "tests/tests.h"

#include "tests/expect.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using namespace tests;
using namespace functional;
using Targets = std::vector<std::int32_t>;

static const std::int32_t IGN = -100;

static void fill_random(Tensor &t, std::mt19937 &rng, float lo = -3.0f,
                        float hi = 3.0f) {
  std::uniform_real_distribution<float> dist(lo, hi);
  for (std::size_t n = 0; n < t.numel(); ++n)
    t.data()[n] = dist(rng);
}

static bool all_finite(const Tensor &t) {
  for (std::size_t n = 0; n < t.numel(); ++n)
    if (!std::isfinite(t.data()[n]))
      return false;
  return true;
}

// Independent reference: the textbook -log(softmax) in double, WITHOUT the
// max trick. Only valid for moderate logits, which is all we feed it.
static double naive_row_loss(const float *z, std::size_t V, std::int32_t t) {
  double sum = 0.0;
  for (std::size_t j = 0; j < V; ++j)
    sum += std::exp(static_cast<double>(z[j]));
  return -std::log(std::exp(static_cast<double>(z[t])) / sum);
}

static double naive_mean_loss(const Tensor &L, const Targets &t) {
  const std::size_t V = L.shape().back();
  double total = 0.0;
  std::size_t count = 0;
  for (std::size_t r = 0; r < t.size(); ++r) {
    if (t[r] == IGN)
      continue;
    total += naive_row_loss(L.data() + r * V, V, t[r]);
    ++count;
  }
  return count ? total / static_cast<double>(count) : 0.0;
}

// =====================================================================
// forward
// =====================================================================

// All-zero logits: every class has probability 1/V, so loss == log(V).
// Gradient: 1/V at non-target entries, (1/V - 1) at the target, over count.
static void test_uniform_logits() {
  const std::size_t V = 10;
  Tensor L({3, V}), dL({3, V});
  Targets t = {0, 5, 9};

  CrossEntropyResult r = cross_entropy(L, t.data());
  EXPECT(r.count == 3);
  EXPECT(std::abs(r.loss - std::log(10.0)) < 1e-12);

  cross_entropy_backward(L, t.data(), dL);
  for (std::size_t row = 0; row < 3; ++row)
    for (std::size_t j = 0; j < V; ++j) {
      const double onehot =
          (static_cast<std::int32_t>(j) == t[row]) ? 1.0 : 0.0;
      const double expected = (0.1 - onehot) / 3.0;
      EXPECT(std::abs(dL.data()[row * V + j] - expected) < 1e-7);
    }
  std::cout << "cross_entropy (uniform logits: loss == log V, gradient): ok\n";
}

// Hand-computed: logits [2,1,0], target 0 -> log(1 + e^-1 + e^-2) = 0.407606;
// target 2 -> 2.407606; mean of the two rows = 1.407606.
static void test_hand_values() {
  Tensor L = Tensor::from_values({2, 3}, {2, 1, 0, 2, 1, 0});
  Targets t = {0, 2};

  CrossEntropyResult r = cross_entropy(L, t.data());
  const double base = std::log(1.0 + std::exp(-1.0) + std::exp(-2.0));
  EXPECT(std::abs(base - 0.40760596) < 1e-7); // sanity on the reference
  EXPECT(std::abs(r.loss - (base + base + 2.0) / 2.0) < 1e-9);
  EXPECT(std::abs(r.loss - 1.40760596) < 1e-7);
  EXPECT(r.count == 2);

  // Gradient row 0: (softmax - onehot(0)) / 2, with softmax = e^z / sum.
  Tensor dL({2, 3});
  cross_entropy_backward(L, t.data(), dL);
  const double s = 1.0 + std::exp(-1.0) + std::exp(-2.0);
  const double p0 = 1.0 / s, p1 = std::exp(-1.0) / s, p2 = std::exp(-2.0) / s;
  EXPECT(std::abs(dL.data()[0] - (p0 - 1.0) / 2.0) < 1e-7);
  EXPECT(std::abs(dL.data()[1] - p1 / 2.0) < 1e-7);
  EXPECT(std::abs(dL.data()[2] - p2 / 2.0) < 1e-7);
  // Row 1 has target 2: (p0, p1, p2 - 1) / 2.
  EXPECT(std::abs(dL.data()[3] - p0 / 2.0) < 1e-7);
  EXPECT(std::abs(dL.data()[5] - (p2 - 1.0) / 2.0) < 1e-7);
  std::cout << "cross_entropy (hand-checked values and gradient): ok\n";
}

static void test_matches_naive_reference() {
  std::mt19937 rng(1);
  Tensor L({6, 9});
  fill_random(L, rng);
  Targets t = {3, 0, IGN, 8, 4, IGN};
  CrossEntropyResult r = cross_entropy(L, t.data());
  EXPECT(r.count == 4);
  EXPECT(std::abs(r.loss - naive_mean_loss(L, t)) < 1e-9);
  std::cout << "cross_entropy (matches naive -log softmax reference): ok\n";
}

// Large logits must not overflow; the loss equals the one for shifted logits.
static void test_stability() {
  Tensor big = Tensor::from_values({1, 3}, {1000, 1001, 1000});
  Targets t = {1};
  CrossEntropyResult r = cross_entropy(big, t.data());
  EXPECT(std::isfinite(r.loss));
  EXPECT(std::abs(r.loss - std::log(1.0 + 2.0 * std::exp(-1.0))) < 1e-9);

  Tensor dL({1, 3});
  cross_entropy_backward(big, t.data(), dL);
  EXPECT(all_finite(dL));

  // Very negative logits: still finite, uniform -> log 3.
  Tensor neg = Tensor::from_values({1, 3}, {-1000, -1000, -1000});
  CrossEntropyResult rn = cross_entropy(neg, t.data());
  EXPECT(std::abs(rn.loss - std::log(3.0)) < 1e-9);
  std::cout << "cross_entropy (numerical stability): ok\n";
}

// Adding a constant to a row's logits changes neither loss nor gradient.
static void test_shift_invariance() {
  std::mt19937 rng(2);
  Tensor L({4, 7}), S({4, 7});
  fill_random(L, rng);
  for (std::size_t r = 0; r < 4; ++r)
    for (std::size_t j = 0; j < 7; ++j)
      S.data()[r * 7 + j] = L.data()[r * 7 + j] + 3.5f * (r + 1);
  Targets t = {0, 6, 3, 2};

  CrossEntropyResult a = cross_entropy(L, t.data());
  CrossEntropyResult b = cross_entropy(S, t.data());
  EXPECT(std::abs(a.loss - b.loss) < 1e-5);

  Tensor dA({4, 7}), dB({4, 7});
  cross_entropy_backward(L, t.data(), dA);
  cross_entropy_backward(S, t.data(), dB);
  for (std::size_t n = 0; n < dA.numel(); ++n)
    EXPECT(std::abs(dA.data()[n] - dB.data()[n]) < 1e-5f);
  std::cout << "cross_entropy (shift invariance): ok\n";
}

static void test_rank3_matches_flat() {
  std::mt19937 rng(3);
  Tensor L3({2, 3, 8}), L2({6, 8});
  fill_random(L3, rng);
  for (std::size_t n = 0; n < L3.numel(); ++n)
    L2.data()[n] = L3.data()[n];
  Targets t = {1, 7, IGN, 0, 3, 5};

  CrossEntropyResult a = cross_entropy(L3, t.data());
  CrossEntropyResult b = cross_entropy(L2, t.data());
  EXPECT(a.count == b.count && a.loss == b.loss);

  Tensor d3({2, 3, 8}), d2({6, 8});
  cross_entropy_backward(L3, t.data(), d3);
  cross_entropy_backward(L2, t.data(), d2);
  for (std::size_t n = 0; n < d3.numel(); ++n)
    EXPECT(d3.data()[n] == d2.data()[n]);
  std::cout << "cross_entropy (rank 3 == flattened rank 2): ok\n";
}

// =====================================================================
// padding / ignore index
// =====================================================================

// A padded batch must behave exactly like the same batch with the padded
// rows removed: same loss, same count, same gradient on the kept rows, and
// exactly zero gradient on the ignored rows (even if dlogits held stale data).
static void test_ignore_index() {
  std::mt19937 rng(4);
  const std::size_t V = 5;
  Tensor full({4, V});
  fill_random(full, rng);
  Targets t_full = {2, IGN, 4, IGN};

  Tensor compact({2, V}); // rows 0 and 2 of `full`
  for (std::size_t j = 0; j < V; ++j) {
    compact.data()[0 * V + j] = full.data()[0 * V + j];
    compact.data()[1 * V + j] = full.data()[2 * V + j];
  }
  Targets t_compact = {2, 4};

  CrossEntropyResult a = cross_entropy(full, t_full.data());
  CrossEntropyResult b = cross_entropy(compact, t_compact.data());
  EXPECT(a.count == 2 && b.count == 2);
  EXPECT(std::abs(a.loss - b.loss) < 1e-12);

  Tensor d_full({4, V}), d_compact({2, V});
  d_full.fill(99.0f); // stale values must be overwritten
  cross_entropy_backward(full, t_full.data(), d_full);
  cross_entropy_backward(compact, t_compact.data(), d_compact);

  for (std::size_t j = 0; j < V; ++j) {
    EXPECT(d_full.data()[1 * V + j] == 0.0f); // ignored rows: exactly zero
    EXPECT(d_full.data()[3 * V + j] == 0.0f);
    EXPECT(std::abs(d_full.data()[0 * V + j] - d_compact.data()[0 * V + j]) <
           1e-7f);
    EXPECT(std::abs(d_full.data()[2 * V + j] - d_compact.data()[1 * V + j]) <
           1e-7f);
  }
  std::cout << "cross_entropy (ignored rows: loss, count, zero gradient): ok\n";
}

// A custom ignore index (e.g. a tokenizer's pad id) must be honoured.
static void test_custom_ignore_index() {
  std::mt19937 rng(5);
  Tensor L({3, 6});
  fill_random(L, rng);
  Targets t = {1, 7777, 4}; // 7777 plays the role of a padding marker

  // Out-of-range value, but it is the ignore index, so it must never be read.
  CrossEntropyResult r = cross_entropy(L, t.data(), /*ignore_index=*/7777);
  EXPECT(r.count == 2);

  Tensor d({3, 6});
  d.fill(1.0f);
  cross_entropy_backward(L, t.data(), d, 7777);
  for (std::size_t j = 0; j < 6; ++j)
    EXPECT(d.data()[1 * 6 + j] == 0.0f);
  std::cout << "cross_entropy (custom ignore index): ok\n";
}

// Nothing valid: loss 0, count 0, zero gradient, no NaN.
static void test_all_ignored() {
  std::mt19937 rng(6);
  Tensor L({3, 4});
  fill_random(L, rng);
  Targets t = {IGN, IGN, IGN};

  CrossEntropyResult r = cross_entropy(L, t.data());
  EXPECT(r.count == 0);
  EXPECT(r.loss == 0.0);

  Tensor d({3, 4});
  d.fill(99.0f);
  cross_entropy_backward(L, t.data(), d);
  for (std::size_t n = 0; n < d.numel(); ++n)
    EXPECT(d.data()[n] == 0.0f);

  // In accumulate mode, an all-ignored batch must leave dlogits untouched.
  d.fill(5.0f);
  cross_entropy_backward(L, t.data(), d, IGN, /*accumulate=*/true);
  for (std::size_t n = 0; n < d.numel(); ++n)
    EXPECT(d.data()[n] == 5.0f);
  std::cout << "cross_entropy (all rows ignored: loss 0, no NaN): ok\n";
}

// =====================================================================
// backward
// =====================================================================

// softmax - onehot sums to zero over each row, so every valid row of the
// gradient sums to ~0 (and ignored rows are exactly zero).
static void test_gradient_row_sums() {
  std::mt19937 rng(7);
  Tensor L({5, 7}), d({5, 7});
  fill_random(L, rng);
  Targets t = {0, 6, IGN, 3, 2};
  cross_entropy_backward(L, t.data(), d);
  for (std::size_t r = 0; r < 5; ++r) {
    double sum = 0.0;
    for (std::size_t j = 0; j < 7; ++j)
      sum += d.data()[r * 7 + j];
    EXPECT(std::abs(sum) < 1e-6);
  }
  std::cout << "cross_entropy backward (rows sum to zero): ok\n";
}

static void test_accumulate() {
  std::mt19937 rng(8);
  Tensor L({4, 6});
  fill_random(L, rng);
  Targets t = {1, IGN, 5, 0};

  Tensor ref({4, 6});
  cross_entropy_backward(L, t.data(), ref);

  Tensor d({4, 6});
  d.fill(10.0f);
  cross_entropy_backward(L, t.data(), d, IGN, /*accumulate=*/true);
  for (std::size_t n = 0; n < d.numel(); ++n)
    EXPECT(std::abs(d.data()[n] - (ref.data()[n] + 10.0f)) < 1e-6f);

  d.fill(10.0f);
  cross_entropy_backward(L, t.data(), d, IGN, /*accumulate=*/false);
  for (std::size_t n = 0; n < d.numel(); ++n)
    EXPECT(d.data()[n] == ref.data()[n]);
  std::cout << "cross_entropy backward (accumulate vs overwrite): ok\n";
}

// The fused gradient must equal what the separate Softmax::backward gives
// when fed the gradient of  -log(p_t) / count  with respect to p.
static void test_matches_softmax_path() {
  std::mt19937 rng(9);
  const std::size_t V = 8;
  Tensor L({5, V});
  fill_random(L, rng);
  Targets t = {2, 7, IGN, 0, 4};

  Softmax sm;
  Tensor P({5, V});
  sm.forward(L, P);

  const double count = 4.0;
  Tensor dP({5, V}); // zero-initialized
  for (std::size_t r = 0; r < 5; ++r)
    if (t[r] != IGN)
      dP.data()[r * V + t[r]] = static_cast<float>(
          -1.0 / (count * static_cast<double>(P.data()[r * V + t[r]])));

  Tensor via_softmax({5, V});
  sm.backward(L, P, dP, via_softmax, false);

  Tensor fused({5, V});
  cross_entropy_backward(L, t.data(), fused);
  for (std::size_t n = 0; n < fused.numel(); ++n)
    EXPECT(std::abs(fused.data()[n] - via_softmax.data()[n]) < 1e-6f);
  std::cout << "cross_entropy backward (== Softmax::backward path): ok\n";
}

// Numeric gradient of the REAL scalar loss. The loss is accumulated in
// double, so a small eps and tight tolerance are safe here.
static void test_gradcheck() {
  std::mt19937 rng(10);
  Tensor L({5, 6}), d({5, 6});
  fill_random(L, rng);
  Targets t = {1, IGN, 3, 0, IGN};

  auto loss = [&]() { return cross_entropy(L, t.data()).loss; };

  cross_entropy_backward(L, t.data(), d);

  const double eps = 1e-3, tol = 1e-3;
  GradcheckResult r = check_gradient(L, d, loss, eps, tol);
  std::cout << "cross_entropy gradcheck: max rel err " << r.max_rel_error
            << "\n";
  EXPECT(r.passed);

  // Self-test 1: scaled gradient must FAIL.
  Tensor d_bad = d.clone();
  for (std::size_t n = 0; n < d_bad.numel(); ++n)
    d_bad.data()[n] *= 1.1f;
  EXPECT(!check_gradient(L, d_bad, loss, eps, tol).passed);

  // Self-test 2: dividing by the TOTAL row count (5) instead of the valid
  // count (3) is the classic normalization bug. It must FAIL.
  Tensor d_norm = d.clone();
  for (std::size_t n = 0; n < d_norm.numel(); ++n)
    d_norm.data()[n] *= 3.0f / 5.0f;
  EXPECT(!check_gradient(L, d_norm, loss, eps, tol).passed);

  std::cout << "cross_entropy gradcheck (incl. self-tests): ok\n";
}

// =====================================================================
// Tensor factories
// =====================================================================

static void test_tensor_factories() {
  // initializer_list overload
  Tensor a = Tensor::from_values({2, 3}, {1, 2, 3, 4, 5, 6});
  EXPECT(a.shape() == std::vector<std::size_t>({2, 3}));
  EXPECT(a(0, 0) == 1 && a(0, 2) == 3 && a(1, 0) == 4 && a(1, 2) == 6);

  // vector overload, and the result is an independent copy
  std::vector<float> src = {1, 2, 3, 4};
  Tensor b = Tensor::from_values({2, 2}, src);
  src[0] = 99.0f;
  EXPECT(b(0, 0) == 1.0f);
  EXPECT(b(1, 1) == 4.0f);

  // pointer overload
  const float raw[3] = {7, 8, 9};
  Tensor c = Tensor::from_pointer({3}, raw, 3);
  EXPECT(c(0) == 7 && c(1) == 8 && c(2) == 9);

  // count mismatch must throw (vector overload)
  bool threw = false;
  try {
    Tensor::from_values({2, 2}, std::vector<float>{1, 2, 3});
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  EXPECT(threw);
  std::cout
      << "tensor factories (values, copy independence, pointer, throw): ok\n";
}

void tests::run_cross_entropy() {
  test_tensor_factories();

  test_uniform_logits();
  test_hand_values();
  test_matches_naive_reference();
  test_stability();
  test_shift_invariance();
  test_rank3_matches_flat();

  test_ignore_index();
  test_custom_ignore_index();
  test_all_ignored();

  test_gradient_row_sums();
  test_accumulate();
  test_matches_softmax_path();
  test_gradcheck();

  std::cout << "all cross-entropy tests passed\n";
}
