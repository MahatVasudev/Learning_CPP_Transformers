#include "tests/tests.h"
#include <functional/gradcheck.h>
#include <functional/module.h> // adjust: wherever Embedding is declared
#include <functional/ops.h>    // gemm
#include <functional/tensor.h>

#include "tests/expect.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

using namespace tests;
using namespace functional;
using Ids = std::vector<std::int32_t>;

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

static bool same_data(const Tensor &a, const Tensor &b) {
  if (a.shape() != b.shape())
    return false;
  for (std::size_t i = 0; i < a.numel(); ++i)
    if (a.data()[i] != b.data()[i])
      return false;
  return true;
}

static bool row_is_zero(const Tensor &t, std::size_t row, std::size_t width) {
  for (std::size_t j = 0; j < width; ++j)
    if (t.data()[row * width + j] != 0.0f)
      return false;
  return true;
}

// table(r, c) = r * 10 + c, so every entry identifies where it came from.
static void fill_labeled(Tensor &table, std::size_t skip_row = SIZE_MAX) {
  const std::size_t d = table.shape()[1];
  for (std::size_t r = 0; r < table.shape()[0]; ++r) {
    if (r == skip_row)
      continue;
    for (std::size_t c = 0; c < d; ++c)
      table.data()[r * d + c] = static_cast<float>(r * 10 + c);
  }
}

template <typename F> static bool throws_out_of_range(F f) {
  try {
    f();
  } catch (const std::out_of_range &) {
    return true;
  }
  return false;
}

// One-hot oracle: O is (n, vocab); rows of padded ids are left all-zero
// when `skip` is set (that is what "no gradient for padding" means).
static Tensor one_hot(const Ids &ids, std::size_t vocab,
                      std::optional<std::size_t> skip = std::nullopt) {
  Tensor O({ids.size(), vocab});
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const std::size_t id = static_cast<std::size_t>(ids[i]);
    if (skip && id == *skip)
      continue;
    O.data()[i * vocab + id] = 1.0f;
  }
  return O;
}

// =====================================================================
// structure and construction
// =====================================================================

static void test_structure() {
  Embedding emb(10, 4);
  Module &base = emb;
  EXPECT(base.parameters().size() == 1);
  EXPECT(base.gradients().size() == 1);
  const std::vector<std::size_t> shape = {10, 4};
  EXPECT(base.parameters()[0]->shape() == shape);
  EXPECT(base.gradients()[0]->shape() == shape);
  for (std::size_t i = 0; i < base.gradients()[0]->numel(); ++i)
    EXPECT(base.gradients()[0]->data()[i] == 0.0f);
  std::cout << "embedding (structure: one parameter, one gradient): ok\n";
}

static void test_init() {
  Embedding a(50, 8, 7), b(50, 8, 7), c(50, 8, 8);
  EXPECT(same_data(*a.parameters()[0], *b.parameters()[0]));
  EXPECT(!same_data(*a.parameters()[0], *c.parameters()[0]));

  // Large table: sample mean ~0, sample std ~1 (N(0,1) init).
  Embedding big(400, 100, 3);
  const Tensor &t = *big.parameters()[0];
  double sum = 0.0, sq = 0.0;
  for (std::size_t i = 0; i < t.numel(); ++i) {
    sum += t.data()[i];
    sq += static_cast<double>(t.data()[i]) * t.data()[i];
  }
  const double n = static_cast<double>(t.numel());
  const double mean = sum / n;
  EXPECT(std::abs(mean) < 0.02);
  EXPECT(std::abs(std::sqrt(sq / n - mean * mean) - 1.0) < 0.02);
  std::cout << "embedding (seeded init, N(0,1) statistics): ok\n";
}

// Embedding(vocab, d, 42) must mean "seed 42, no padding index" -- the
// argument-order trap from before. No row may be all zeros.
static void test_no_padding_by_default() {
  Embedding emb(100, 8, 42);
  const Tensor &t = *emb.parameters()[0];
  for (std::size_t r = 0; r < 100; ++r)
    EXPECT(!row_is_zero(t, r, 8));
  std::cout
      << "embedding (no padding index by default, seed arg is the seed): ok\n";
}

static void test_padding_row_zero_at_init() {
  Embedding emb(20, 6, 5, /*padding_idx=*/3);
  const Tensor &t = *emb.parameters()[0];
  EXPECT(row_is_zero(t, 3, 6));
  for (std::size_t r = 0; r < 20; ++r)
    if (r != 3)
      EXPECT(!row_is_zero(t, r, 6));
  std::cout << "embedding (padding row zeroed at construction): ok\n";
}

static void test_bad_padding_idx_throws() {
  EXPECT(throws_out_of_range([] { Embedding e(10, 4, 0, 10); }));
  EXPECT(throws_out_of_range([] { Embedding e(10, 4, 0, 99); }));
  std::cout << "embedding (padding_idx out of range throws): ok\n";
}

// =====================================================================
// forward
// =====================================================================

// Includes the LAST token and a repeated id; Y is sentinel-filled so any
// row that is not written shows up as 99.
static void test_forward_hand() {
  Embedding emb(5, 3);
  fill_labeled(*emb.parameters()[0]);

  Ids ids = {2, 0, 2, 3};
  Tensor Y({4, 3});
  Y.fill(99.0f);
  emb.forward(ids.data(), ids.size(), Y);

  const float expected[4][3] = {
      {20, 21, 22}, {0, 1, 2}, {20, 21, 22}, {30, 31, 32}};
  for (std::size_t i = 0; i < 4; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      EXPECT(Y.data()[i * 3 + j] == expected[i][j]);
  std::cout
      << "embedding forward (hand-checked, last token, stale output): ok\n";
}

// A padding id must produce exact zeros even when Y held stale values.
static void test_forward_padding() {
  Embedding emb(5, 3, 1, /*padding_idx=*/1);
  EXPECT(row_is_zero(*emb.parameters()[0], 1, 3));
  fill_labeled(*emb.parameters()[0], /*skip_row=*/1);

  Ids ids = {1, 3, 1};
  Tensor Y({3, 3});
  Y.fill(99.0f);
  emb.forward(ids.data(), ids.size(), Y);

  for (std::size_t j = 0; j < 3; ++j) {
    EXPECT(Y.data()[0 * 3 + j] == 0.0f);
    EXPECT(Y.data()[1 * 3 + j] == static_cast<float>(30 + j));
    EXPECT(Y.data()[2 * 3 + j] == 0.0f);
  }
  std::cout << "embedding forward (padding id gives exact zeros): ok\n";
}

static void test_forward_rank3() {
  Embedding emb(12, 5, 9);
  Ids ids = {3, 11, 0, 7, 7, 2};
  Tensor Y3({2, 3, 5}), Y2({6, 5});
  emb.forward(ids.data(), ids.size(), Y3);
  emb.forward(ids.data(), ids.size(), Y2);
  for (std::size_t i = 0; i < Y3.numel(); ++i)
    EXPECT(Y3.data()[i] == Y2.data()[i]);
  std::cout << "embedding forward (rank 3 == flattened rank 2): ok\n";
}

// n == 0 must not underflow (the old `n - 1` loop bound would).
static void test_empty_batch() {
  Embedding emb(5, 3);
  Ids ids = {0};
  Tensor Y({0, 3}), dY({0, 3});
  emb.forward(ids.data(), 0, Y);
  emb.backward(ids.data(), 0, dY);
  for (std::size_t i = 0; i < emb.gradients()[0]->numel(); ++i)
    EXPECT(emb.gradients()[0]->data()[i] == 0.0f);
  std::cout << "embedding (n == 0 forward and backward do nothing): ok\n";
}

static void test_out_of_range_ids() {
  Embedding emb(5, 3);
  Tensor Y({2, 3}), dY({2, 3});
  Ids neg = {0, -1};
  Ids big = {0, 5}; // == vocab_size
  EXPECT(throws_out_of_range([&] { emb.forward(neg.data(), 2, Y); }));
  EXPECT(throws_out_of_range([&] { emb.forward(big.data(), 2, Y); }));
  EXPECT(throws_out_of_range([&] { emb.backward(neg.data(), 2, dY); }));
  EXPECT(throws_out_of_range([&] { emb.backward(big.data(), 2, dY); }));
  std::cout << "embedding (ids -1 and vocab_size throw out_of_range): ok\n";
}

// =====================================================================
// backward
// =====================================================================

// ids {2,0,2,4,2}, dY row i = (i+1) * [1, 10, 100].
// Row 2 appears three times -> sum of dY rows 0, 2, 4 = [9, 90, 900].
static void test_backward_hand() {
  Embedding emb(5, 3);
  Ids ids = {2, 0, 2, 4, 2};
  Tensor dY({5, 3});
  for (std::size_t i = 0; i < 5; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      dY.data()[i * 3 + j] = static_cast<float>((i + 1) * std::pow(10, j));

  emb.backward(ids.data(), ids.size(), dY);
  const Tensor &g = *emb.gradients()[0];

  const float row0[3] = {2, 20, 200};
  const float row2[3] = {9, 90, 900};
  const float row4[3] = {4, 40, 400};
  for (std::size_t j = 0; j < 3; ++j) {
    EXPECT(g.data()[0 * 3 + j] == row0[j]);
    EXPECT(g.data()[2 * 3 + j] == row2[j]);
    EXPECT(g.data()[4 * 3 + j] == row4[j]);
  }
  EXPECT(row_is_zero(g, 1, 3)); // ids that never appear: exactly zero
  EXPECT(row_is_zero(g, 3, 3));
  std::cout << "embedding backward (hand-checked, repeated ids sum): ok\n";
}

// Independent oracle: forward == O * table, backward == O^T * dY, both via
// gemm on an explicit one-hot matrix.
static void test_one_hot_oracle() {
  std::mt19937 rng(1);
  const std::size_t vocab = 9, d = 4;
  Embedding emb(vocab, d, 2);
  Ids ids = {3, 8, 3, 0, 5, 8, 3};
  const std::size_t n = ids.size();
  Tensor O = one_hot(ids, vocab);

  // forward: Y = O * Emb
  Tensor Y({n, d}), Y_ref({n, d});
  emb.forward(ids.data(), n, Y);
  gemm(O.data(), emb.parameters()[0]->data(), Y_ref.data(), n, vocab, d,
       /*lda=*/vocab, 1, /*ldb=*/d, 1, /*ldc=*/d, 1, false, false, false);
  for (std::size_t i = 0; i < Y.numel(); ++i)
    EXPECT(std::abs(Y.data()[i] - Y_ref.data()[i]) < 1e-6f);

  // backward: dEmb = O^T * dY
  Tensor dY({n, d});
  fill_random(dY, rng);
  emb.backward(ids.data(), n, dY);
  Tensor g_ref({vocab, d});
  gemm(O.data(), dY.data(), g_ref.data(), vocab, n, d,
       /*lda=*/vocab, 1, /*ldb=*/d, 1, /*ldc=*/d, 1, /*transA=*/true, false,
       false);
  const Tensor &g = *emb.gradients()[0];
  for (std::size_t i = 0; i < g.numel(); ++i)
    EXPECT(std::abs(g.data()[i] - g_ref.data()[i]) < 1e-6f);
  std::cout << "embedding (one-hot gemm oracle, forward and backward): ok\n";
}

// Padded positions contribute no gradient: oracle with padded rows zeroed,
// and the padding row of the gradient stays exactly zero.
static void test_backward_padding() {
  std::mt19937 rng(2);
  const std::size_t vocab = 6, d = 4, pad = 1;
  Embedding emb(vocab, d, 3, pad);
  Ids ids = {1, 2, 1, 3, 2, 1};
  const std::size_t n = ids.size();

  Tensor dY({n, d});
  fill_random(dY, rng);
  emb.backward(ids.data(), n, dY);
  const Tensor &g = *emb.gradients()[0];
  EXPECT(row_is_zero(g, pad, d));

  Tensor O = one_hot(ids, vocab, pad);
  Tensor g_ref({vocab, d});
  gemm(O.data(), dY.data(), g_ref.data(), vocab, n, d, vocab, 1, d, 1, d, 1,
       true, false, false);
  for (std::size_t i = 0; i < g.numel(); ++i)
    EXPECT(std::abs(g.data()[i] - g_ref.data()[i]) < 1e-6f);
  std::cout << "embedding backward (padding row gets no gradient): ok\n";
}

// backward must only touch the gradient, never the table itself.
static void test_backward_leaves_table_alone() {
  std::mt19937 rng(3);
  Embedding emb(7, 3, 4);
  Tensor before = emb.parameters()[0]->clone();
  Ids ids = {1, 6, 1};
  Tensor dY({3, 3});
  fill_random(dY, rng);
  emb.backward(ids.data(), 3, dY);
  EXPECT(same_data(before, *emb.parameters()[0]));
  std::cout << "embedding backward (table unchanged): ok\n";
}

static void test_accumulation_and_zero_grad() {
  std::mt19937 rng(4);
  Embedding emb(6, 3, 5);
  Ids ids = {4, 4, 0};
  Tensor dY({3, 3});
  fill_random(dY, rng);

  emb.backward(ids.data(), 3, dY);
  Tensor once = emb.gradients()[0]->clone();
  emb.backward(ids.data(), 3, dY); // no zero_grad in between
  const Tensor &g = *emb.gradients()[0];
  for (std::size_t i = 0; i < g.numel(); ++i)
    EXPECT(std::abs(g.data()[i] - 2.0f * once.data()[i]) < 1e-6f);

  Module &base = emb;
  base.zero_grad();
  for (std::size_t i = 0; i < g.numel(); ++i)
    EXPECT(g.data()[i] == 0.0f);
  std::cout << "embedding backward (accumulates, zero_grad via Module&): ok\n";
}

// =====================================================================
// gradcheck
// =====================================================================

// No padding index here on purpose: with one, forward READS the padding row
// (numeric gradient nonzero) while backward deliberately writes nothing to it
// (analytic gradient zero), so the two would legitimately disagree.
static void test_gradcheck() {
  std::mt19937 rng(5);
  const std::size_t vocab = 8, d = 5;
  Embedding emb(vocab, d, 6);
  Ids ids = {2, 7, 2, 0, 5, 2};
  const std::size_t n = ids.size();

  Tensor Y({n, d}), R({n, d});
  fill_random(R, rng);

  Tensor &table = *emb.parameters()[0];
  auto loss = [&]() {
    Y.fill(99.0f); // stale values must not survive a forward pass
    emb.forward(ids.data(), n, Y);
    return dot_loss(Y, R);
  };

  emb.zero_grad();
  emb.backward(ids.data(), n, /*dY=*/R);

  const double eps = 1e-2, tol = 1e-3;
  GradcheckResult r =
      check_gradient(table, *emb.gradients()[0], loss, eps, tol);
  std::cout << "embedding gradcheck: max rel err " << r.max_rel_error << "\n";
  EXPECT(r.passed);

  // Self-test: a wrong gradient must FAIL the checker.
  Tensor bad = emb.gradients()[0]->clone();
  for (std::size_t i = 0; i < bad.numel(); ++i)
    bad.data()[i] *= 1.1f;
  EXPECT(!check_gradient(table, bad, loss, eps, tol).passed);

  // Self-test: forgetting that repeated ids SUM (id 2 appears 3 times) must
  // FAIL: simulate by overwriting instead of adding.
  Tensor overwrite({vocab, d});
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < d; ++j)
      overwrite.data()[static_cast<std::size_t>(ids[i]) * d + j] =
          R.data()[i * d + j];
  EXPECT(!check_gradient(table, overwrite, loss, eps, tol).passed);
  std::cout << "embedding gradcheck (incl. self-tests): ok\n";
}

void tests::run_embedding() {
  test_structure();
  test_init();
  test_no_padding_by_default();
  test_padding_row_zero_at_init();
  test_bad_padding_idx_throws();

  test_forward_hand();
  test_forward_padding();
  test_forward_rank3();
  test_empty_batch();
  test_out_of_range_ids();

  test_backward_hand();
  test_one_hot_oracle();
  test_backward_padding();
  test_backward_leaves_table_alone();
  test_accumulation_and_zero_grad();

  test_gradcheck();

  std::cout << "all embedding tests passed\n";
}
