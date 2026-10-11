#include <functional/positional_encoding.h>
#include <functional/tensor.h>
#include <tests/expect.h>
#include <tests/tests.h>

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {

using functional::PositionalEncodingSinusoidal;
using functional::Tensor;

// One place to adapt to your accessor. If encoding() returns a reference
// instead of a pointer, drop the `*`.
const Tensor &table_of(const PositionalEncodingSinusoidal &pe) {
  return *pe.encoding();
}

template <typename F> bool throws_invalid_argument(F f) {
  try {
    f();
  } catch (const std::invalid_argument &) {
    return true;
  }
  return false;
}

bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

// =====================================================================
// the table itself
// =====================================================================

// Position 0: every angle is 0, so the row is [0, 1, 0, 1, ...].
void test_position_zero() {
  PositionalEncodingSinusoidal pe(10, 8);
  const Tensor &t = table_of(pe);
  for (std::size_t j = 0; j < 8; ++j)
    EXPECT(t(0, j) == (j % 2 == 0 ? 0.0f : 1.0f));
  std::cout << "positional encoding (position 0 is [0,1,0,1,...]): ok\n";
}

// d_model = 4, position 1:
//   pair 0: exponent 0   -> angle 1
//   pair 1: exponent 0.5 -> 10000^0.5 = 100 -> angle 0.01
// The doubled-exponent bug gives cos(0.01) in column 1 and fails here.
void test_hand_values() {
  PositionalEncodingSinusoidal pe(4, 4);
  const Tensor &t = table_of(pe);
  EXPECT(near(t(1, 0), 0.8414710, 1e-6)); // sin 1
  EXPECT(near(t(1, 1), 0.5403023, 1e-6)); // cos 1
  EXPECT(near(t(1, 2), 0.0099998, 1e-6)); // sin 0.01
  EXPECT(near(t(1, 3), 0.9999500, 1e-6)); // cos 0.01
  // position 2 doubles every angle
  EXPECT(near(t(2, 0), std::sin(2.0), 1e-6));
  EXPECT(near(t(2, 1), std::cos(2.0), 1e-6));
  EXPECT(near(t(2, 2), std::sin(0.02), 1e-6));
  EXPECT(near(t(2, 3), std::cos(0.02), 1e-6));
  std::cout << "positional encoding (hand-checked values): ok\n";
}

// d_model = 8: pair 3 has exponent 6/8 = 0.75 -> 10000^0.75 = 1000.
void test_slowest_frequency() {
  PositionalEncodingSinusoidal pe(8, 8);
  const Tensor &t = table_of(pe);
  EXPECT(near(t(7, 6), std::sin(0.007), 1e-6));
  EXPECT(near(t(7, 7), std::cos(0.007), 1e-6));
  std::cout << "positional encoding (slowest pair uses 10000^0.75): ok\n";
}

// Whole table against an independent reference written straight from the
// textbook formula.
void test_full_table_reference() {
  const std::size_t L = 20, D = 16;
  PositionalEncodingSinusoidal pe(L, D);
  const Tensor &t = table_of(pe);
  EXPECT(t.shape() == std::vector<std::size_t>({L, D}));
  for (std::size_t pos = 0; pos < L; ++pos)
    for (std::size_t k = 0; k < D / 2; ++k) {
      const double a = static_cast<double>(pos) /
                       std::pow(10000.0, 2.0 * static_cast<double>(k) / D);
      EXPECT(near(t(pos, 2 * k), std::sin(a), 1e-6));
      EXPECT(near(t(pos, 2 * k + 1), std::cos(a), 1e-6));
    }
  std::cout << "positional encoding (full table vs textbook formula): ok\n";
}

// Finite, within [-1, 1], and sin^2 + cos^2 = 1 for every pair (which only
// holds if the two columns of a pair share one angle).
void test_range_and_pairs() {
  PositionalEncodingSinusoidal pe(64, 32);
  const Tensor &t = table_of(pe);
  for (std::size_t pos = 0; pos < 64; ++pos)
    for (std::size_t k = 0; k < 16; ++k) {
      const double s = t(pos, 2 * k), c = t(pos, 2 * k + 1);
      EXPECT(std::isfinite(s) && std::isfinite(c));
      EXPECT(std::abs(s) <= 1.0 && std::abs(c) <= 1.0);
      EXPECT(near(s * s + c * c, 1.0, 1e-6));
    }
  std::cout
      << "positional encoding (finite, in range, pairs share angle): ok\n";
}

void test_odd_d_model_throws() {
  EXPECT(
      throws_invalid_argument([] { PositionalEncodingSinusoidal pe(10, 7); }));
  std::cout << "positional encoding (odd d_model throws): ok\n";
}

// =====================================================================
// add_to
// =====================================================================

// (2, 3, 4) against max_len = 8, filled with 100. Needs a sequence SHORTER
// than max_len, a batch of 2 (second sequence must restart at position 0),
// and seq_len != d_model. Hits the length check, the row % seq_len position,
// and the inner-loop bound all at once.
void test_add_to_rank3() {
  PositionalEncodingSinusoidal pe(8, 4);
  const Tensor &tbl = table_of(pe);

  Tensor X({2, 3, 4});
  X.fill(100.0f);
  pe.add_to(X);

  for (std::size_t b = 0; b < 2; ++b)
    for (std::size_t s = 0; s < 3; ++s)
      for (std::size_t j = 0; j < 4; ++j)
        EXPECT(X(b, s, j) == 100.0f + tbl(s, j));
  std::cout << "positional encoding add_to (rank 3, short seq, batch 2): ok\n";
}

// seq_len > d_model: a loop bounded by seq_len would write past the end of
// each row. (Also a sanitizer test.)
void test_add_to_seq_longer_than_d_model() {
  PositionalEncodingSinusoidal pe(8, 4);
  const Tensor &tbl = table_of(pe);

  Tensor X({2, 6, 4});
  X.fill(-5.0f);
  pe.add_to(X);

  for (std::size_t b = 0; b < 2; ++b)
    for (std::size_t s = 0; s < 6; ++s)
      for (std::size_t j = 0; j < 4; ++j)
        EXPECT(X(b, s, j) == -5.0f + tbl(s, j));
  std::cout << "positional encoding add_to (seq_len > d_model): ok\n";
}

void test_add_to_rank2() {
  PositionalEncodingSinusoidal pe(8, 4);
  const Tensor &tbl = table_of(pe);
  Tensor X({3, 4});
  X.fill(1.0f);
  pe.add_to(X);
  for (std::size_t s = 0; s < 3; ++s)
    for (std::size_t j = 0; j < 4; ++j)
      EXPECT(X(s, j) == 1.0f + tbl(s, j));
  std::cout << "positional encoding add_to (rank 2 = one sequence): ok\n";
}

// Rank 4: the sequence is always the second-to-last dimension.
void test_add_to_rank4() {
  PositionalEncodingSinusoidal pe(8, 4);
  const Tensor &tbl = table_of(pe);
  Tensor X({2, 2, 3, 4});
  X.fill(0.0f);
  pe.add_to(X);
  for (std::size_t a = 0; a < 2; ++a)
    for (std::size_t b = 0; b < 2; ++b)
      for (std::size_t s = 0; s < 3; ++s)
        for (std::size_t j = 0; j < 4; ++j)
          EXPECT(X(a, b, s, j) == tbl(s, j));
  std::cout << "positional encoding add_to (rank 4): ok\n";
}

// seq_len == max_len is allowed (the check is >, not >=).
void test_add_to_full_length() {
  PositionalEncodingSinusoidal pe(5, 4);
  const Tensor &tbl = table_of(pe);
  Tensor X({2, 5, 4});
  pe.add_to(X);
  for (std::size_t b = 0; b < 2; ++b)
    for (std::size_t s = 0; s < 5; ++s)
      for (std::size_t j = 0; j < 4; ++j)
        EXPECT(X(b, s, j) == tbl(s, j));
  std::cout << "positional encoding add_to (seq_len == max_len): ok\n";
}

// It must ADD to what is there, not overwrite it.
void test_add_to_adds() {
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  PositionalEncodingSinusoidal pe(8, 6);
  const Tensor &tbl = table_of(pe);

  Tensor X({3, 4, 6}), orig({3, 4, 6});
  for (std::size_t i = 0; i < X.numel(); ++i)
    orig.data()[i] = X.data()[i] = dist(rng);
  pe.add_to(X);
  for (std::size_t b = 0; b < 3; ++b)
    for (std::size_t s = 0; s < 4; ++s)
      for (std::size_t j = 0; j < 6; ++j)
        EXPECT(X(b, s, j) == orig(b, s, j) + tbl(s, j));
  std::cout << "positional encoding add_to (adds, does not overwrite): ok\n";
}

// add_to must never modify the table, and applying it twice adds it twice.
void test_table_unchanged() {
  PositionalEncodingSinusoidal pe(8, 4);
  Tensor before = table_of(pe).clone();
  Tensor X({2, 3, 4});
  pe.add_to(X);
  pe.add_to(X);
  const Tensor &after = table_of(pe);
  for (std::size_t i = 0; i < before.numel(); ++i)
    EXPECT(before.data()[i] == after.data()[i]);
  for (std::size_t s = 0; s < 3; ++s)
    for (std::size_t j = 0; j < 4; ++j)
      EXPECT(X(1, s, j) == after(s, j) + after(s, j));
  std::cout << "positional encoding (table unchanged by add_to): ok\n";
}

void test_add_to_misuse() {
  PositionalEncodingSinusoidal pe(8, 4);
  Tensor too_long({1, 9, 4}); // seq_len 9 > max_len 8
  Tensor wrong_d({2, 3, 5});  // last dim != d_model
  Tensor rank1({4});
  EXPECT(throws_invalid_argument([&] { pe.add_to(too_long); }));
  EXPECT(throws_invalid_argument([&] { pe.add_to(wrong_d); }));
  EXPECT(throws_invalid_argument([&] { pe.add_to(rank1); }));
  std::cout << "positional encoding add_to (misuse throws): ok\n";
}

} // namespace

namespace tests {
void run_positional_encoding() {
  test_position_zero();
  test_hand_values();
  test_slowest_frequency();
  test_full_table_reference();
  test_range_and_pairs();
  test_odd_d_model_throws();

  test_add_to_rank3();
  test_add_to_seq_longer_than_d_model();
  test_add_to_rank2();
  test_add_to_rank4();
  test_add_to_full_length();
  test_add_to_adds();
  test_table_unchanged();
  test_add_to_misuse();

  std::cout << "all positional encoding tests passed\n";
}
} // namespace tests
