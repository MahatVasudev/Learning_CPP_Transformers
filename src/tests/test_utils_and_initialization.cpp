#include "tests/expect.h"
#include "tests/tests.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional/tensor.h>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <utils/utils.h>
#include <vector>

using namespace tests;
using functional::Tensor;
using utils::fill_random;
using utils::Initialize;

struct Stats {
  double mean, stddev, min, max;
};

static Stats stats(const Tensor &t) {
  const double n = static_cast<double>(t.numel());
  double sum = 0.0, lo = t.data()[0], hi = t.data()[0];
  for (std::size_t i = 0; i < t.numel(); ++i) {
    sum += t.data()[i];
    lo = std::min<double>(lo, t.data()[i]);
    hi = std::max<double>(hi, t.data()[i]);
  }
  const double mean = sum / n;
  double var = 0.0;
  for (std::size_t i = 0; i < t.numel(); ++i) {
    const double c = t.data()[i] - mean;
    var += c * c;
  }
  return {mean, std::sqrt(var / n), lo, hi};
}

static bool same_data(const Tensor &a, const Tensor &b) {
  if (a.shape() != b.shape())
    return false;
  for (std::size_t i = 0; i < a.numel(); ++i)
    if (a.data()[i] != b.data()[i])
      return false;
  return true;
}

static bool within(double value, double target, double rel_tol) {
  return std::abs(value - target) <= rel_tol * std::abs(target);
}

template <typename F> static bool throws_invalid_argument(F f) {
  try {
    f();
  } catch (const std::invalid_argument &) {
    return true;
  }
  return false;
}

// ---------- exact fills ----------

static void test_zeros_and_constant() {
  Tensor t({4, 5});
  t.fill(7.0f); // stale values must be overwritten
  fill_random(t, Initialize::zeros(), 1);
  for (std::size_t i = 0; i < t.numel(); ++i)
    EXPECT(t.data()[i] == 0.0f);

  fill_random(t, Initialize::constant(2.5f), 1);
  for (std::size_t i = 0; i < t.numel(); ++i)
    EXPECT(t.data()[i] == 2.5f);

  // These don't depend on shape, so rank 1 is fine.
  Tensor bias({6});
  fill_random(bias, Initialize::constant(-1.0f), 1);
  for (std::size_t i = 0; i < bias.numel(); ++i)
    EXPECT(bias.data()[i] == -1.0f);
  std::cout << "utils (zeros, constant): ok\n";
}

// ---------- plain distributions ----------

static void test_uniform() {
  Tensor t({200, 200});
  fill_random(t, Initialize::uniform(-2.0f, 6.0f), 11);
  Stats s = stats(t);
  EXPECT(s.min >= -2.0 && s.max <= 6.0);
  EXPECT(s.min < -1.9 && s.max > 5.9); // actually spans the range
  EXPECT(std::abs(s.mean - 2.0) < 0.05);
  EXPECT(within(s.stddev, 8.0 / std::sqrt(12.0), 0.02));
  std::cout << "utils (uniform: bounds, mean, std): ok\n";
}

static void test_normal() {
  Tensor t({200, 200});
  fill_random(t, Initialize::normal(2.0f, 0.5f), 12);
  Stats s = stats(t);
  EXPECT(std::abs(s.mean - 2.0) < 0.02);
  EXPECT(within(s.stddev, 0.5, 0.02));
  std::cout << "utils (normal: mean, std): ok\n";
}

// Guards against the "engine copied on every draw" bug: if the engine never
// advances, every element is identical and there is ~1 distinct value.
static void test_values_are_distinct() {
  Tensor t({100});
  fill_random(t, Initialize::uniform(-1.0f, 1.0f), 13);
  std::set<float> distinct(t.data(), t.data() + t.numel());
  EXPECT(distinct.size() > 90);

  Tensor n({100});
  fill_random(n, Initialize::normal(0.0f, 1.0f), 13);
  std::set<float> distinct_n(n.data(), n.data() + n.numel());
  EXPECT(distinct_n.size() > 90);
  std::cout << "utils (values are distinct, engine advances): ok\n";
}

static void test_seed_behaviour() {
  Tensor a({30, 20}), b({30, 20}), c({30, 20});
  fill_random(a, Initialize::normal(0.0f, 1.0f), 7);
  fill_random(b, Initialize::normal(0.0f, 1.0f), 7);
  fill_random(c, Initialize::normal(0.0f, 1.0f), 8);
  EXPECT(same_data(a, b));
  EXPECT(!same_data(a, c));
  std::cout << "utils (same seed identical, different seed differs): ok\n";
}

// ---------- fan-based kinds ----------
// Shape (300, 200) is deliberately non-square: fan_in = 200, fan_out = 300.
// That separates Xavier (uses both) from Kaiming (fan_in only), and exposes
// the precedence bug  6/fan_in + fan_out  (bound ~300 instead of ~0.11).

static void test_xavier_uniform() {
  Tensor t({300, 200});
  fill_random(t, Initialize::xavier_uniform(), 21);
  const double bound = std::sqrt(6.0 / (200.0 + 300.0));
  Stats s = stats(t);
  EXPECT(s.min >= -bound && s.max <= bound);
  EXPECT(s.max > 0.98 * bound && s.min < -0.98 * bound); // fills the range
  EXPECT(std::abs(s.mean) < 0.003);
  EXPECT(within(s.stddev, bound / std::sqrt(3.0), 0.02));
  std::cout << "utils (xavier uniform: bound, std): ok\n";
}

static void test_xavier_normal() {
  Tensor t({300, 200});
  fill_random(t, Initialize::xavier_normal(), 22);
  Stats s = stats(t);
  EXPECT(std::abs(s.mean) < 0.003);
  EXPECT(within(s.stddev, std::sqrt(2.0 / (200.0 + 300.0)), 0.02));
  std::cout << "utils (xavier normal: std): ok\n";
}

static void test_kaiming_uniform() {
  Tensor t({300, 200});
  fill_random(t, Initialize::kaiming_uniform(), 23);
  const double bound = std::sqrt(6.0 / 200.0); // fan_in only
  Stats s = stats(t);
  EXPECT(s.min >= -bound && s.max <= bound);
  EXPECT(s.max > 0.98 * bound && s.min < -0.98 * bound);
  EXPECT(within(s.stddev, bound / std::sqrt(3.0), 0.02));
  std::cout << "utils (kaiming uniform: bound uses fan_in only): ok\n";
}

static void test_kaiming_normal() {
  Tensor t({300, 200});
  fill_random(t, Initialize::kaiming_normal(), 24);
  Stats s = stats(t);
  EXPECT(std::abs(s.mean) < 0.004);
  EXPECT(within(s.stddev, std::sqrt(2.0 / 200.0), 0.02));
  std::cout << "utils (kaiming normal: std uses fan_in only): ok\n";
}

// The uniform and normal versions of a family target the same variance.
static void test_family_variance_match() {
  Tensor xu({300, 200}), xn({300, 200}), ku({300, 200}), kn({300, 200});
  fill_random(xu, Initialize::xavier_uniform(), 31);
  fill_random(xn, Initialize::xavier_normal(), 32);
  fill_random(ku, Initialize::kaiming_uniform(), 33);
  fill_random(kn, Initialize::kaiming_normal(), 34);
  EXPECT(within(stats(xu).stddev, stats(xn).stddev, 0.03));
  EXPECT(within(stats(ku).stddev, stats(kn).stddev, 0.03));
  std::cout << "utils (uniform vs normal variance per family): ok\n";
}

// Higher rank: dims after the first two multiply BOTH fans.
// (64, 32, 3, 3): fan_in = 32*9 = 288, fan_out = 64*9 = 576.
static void test_rank4_fans() {
  Tensor t({64, 32, 3, 3});
  fill_random(t, Initialize::kaiming_normal(), 41);
  EXPECT(within(stats(t).stddev, std::sqrt(2.0 / 288.0), 0.03));

  Tensor u({64, 32, 3, 3});
  fill_random(u, Initialize::xavier_normal(), 42);
  EXPECT(within(stats(u).stddev, std::sqrt(2.0 / (288.0 + 576.0)), 0.03));
  std::cout << "utils (rank 4: receptive field multiplies fans): ok\n";
}

// Rank 2 must match the exact formula used by the old hand-written Linear
// init, so replacing that loop changes no values.
static void test_matches_old_linear_init() {
  const std::size_t in = 40, out = 50;
  Tensor via_utils({out, in}), by_hand({out, in});
  fill_random(via_utils, Initialize::xavier_uniform(), 5);

  std::mt19937 rng(5);
  const float bound = std::sqrt(6.0f / static_cast<float>(in + out));
  std::uniform_real_distribution<float> dist(-bound, bound);
  for (std::size_t i = 0; i < by_hand.numel(); ++i)
    by_hand.data()[i] = dist(rng);

  EXPECT(same_data(via_utils, by_hand));
  std::cout << "utils (xavier uniform == old Linear init, same seed): ok\n";
}

// ---------- misuse ----------

// Fan-based kinds make no sense for a bias-shaped (rank 1) tensor.
// NOTE: this needs get_fans to THROW, not just assert, so it also works
// in release builds.
static void test_rank1_throws() {
  Tensor bias({16});
  EXPECT(throws_invalid_argument(
      [&] { fill_random(bias, Initialize::xavier_uniform(), 1); }));
  EXPECT(throws_invalid_argument(
      [&] { fill_random(bias, Initialize::xavier_normal(), 1); }));
  EXPECT(throws_invalid_argument(
      [&] { fill_random(bias, Initialize::kaiming_uniform(), 1); }));
  EXPECT(throws_invalid_argument(
      [&] { fill_random(bias, Initialize::kaiming_normal(), 1); }));
  std::cout << "utils (fan-based kinds throw on rank 1): ok\n";
}

void tests::run_utils() {
  test_zeros_and_constant();
  test_uniform();
  test_normal();
  test_values_are_distinct();
  test_seed_behaviour();

  test_xavier_uniform();
  test_xavier_normal();
  test_kaiming_uniform();
  test_kaiming_normal();
  test_family_variance_match();
  test_rank4_fans();
  test_matches_old_linear_init();

  test_rank1_throws();

  std::cout << "all utils tests passed\n";
}
