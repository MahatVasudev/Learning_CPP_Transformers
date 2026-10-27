#include <functional/gradcheck.h>
#include <functional/module.h>
#include <functional/tensor.h>
#include <model/model.h>
#include <tests/expect.h>
#include <tests/tests.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

using functional::GradcheckResult;
using functional::Linear;
using functional::Module;
using functional::ReLU;
using functional::Tensor;
using model::FeedForward;

// ---------- helpers ----------

void fill_random(Tensor &t, std::mt19937 &rng, float lo = -1.0f,
                 float hi = 1.0f) {
  std::uniform_real_distribution<float> dist(lo, hi);
  for (std::size_t n = 0; n < t.numel(); ++n)
    t.data()[n] = dist(rng);
}

void set_values(Tensor &t, const std::vector<float> &vals) {
  EXPECT(t.numel() == vals.size());
  for (std::size_t n = 0; n < vals.size(); ++n)
    t.data()[n] = vals[n];
}

void copy_into(Tensor &dst, const Tensor &src) {
  EXPECT(dst.shape() == src.shape());
  for (std::size_t n = 0; n < src.numel(); ++n)
    dst.data()[n] = src.data()[n];
}

bool same_data(const Tensor &a, const Tensor &b, float tol = 0.0f) {
  if (a.shape() != b.shape())
    return false;
  for (std::size_t n = 0; n < a.numel(); ++n)
    if (std::abs(a.data()[n] - b.data()[n]) > tol)
      return false;
  return true;
}

double dot_loss(const Tensor &Y, const Tensor &R) {
  double L = 0.0;
  for (std::size_t n = 0; n < Y.numel(); ++n)
    L += static_cast<double>(Y.data()[n]) * R.data()[n];
  return L;
}

template <typename Ex, typename F> bool throws(F f) {
  try {
    f();
  } catch (const Ex &) {
    return true;
  }
  return false;
}

// Parameter order is {W1 (d_ff x d_model), b1 (d_ff), W2 (d_model x d_ff), b2
// (d_model)}. Hand-composed reference built from separate layers, using the
// block's own weights.
struct Reference {
  Linear l1, l2;
  ReLU relu;
  Reference(std::size_t d_model, std::size_t d_ff, FeedForward &ff)
      : l1(d_model, d_ff, true, 0), l2(d_ff, d_model, true, 0) {
    auto p = ff.parameters();
    copy_into(*l1.parameters()[0], *p[0]);
    copy_into(*l1.parameters()[1], *p[1]);
    copy_into(*l2.parameters()[0], *p[2]);
    copy_into(*l2.parameters()[1], *p[3]);
  }
};

std::vector<std::size_t> hidden_of(const Tensor &X, std::size_t d_ff) {
  std::vector<std::size_t> s = X.shape();
  s.back() = d_ff;
  return s;
}

// =====================================================================
// structure
// =====================================================================

void test_structure() {
  FeedForward ff(4, 8, 1);
  Module &base = ff;
  auto p = base.parameters();
  auto g = base.gradients();
  EXPECT(p.size() == 4 && g.size() == 4);

  const std::vector<std::vector<std::size_t>> shapes = {
      {8, 4}, {8}, {4, 8}, {4}};
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT(p[i]->shape() == shapes[i]);
    EXPECT(g[i]->shape() == shapes[i]);
  }

  // A weight and its gradient have the same shape, so shape checks alone
  // cannot catch gradients() returning the parameters. Pointers can.
  std::set<Tensor *> all(p.begin(), p.end());
  all.insert(g.begin(), g.end());
  EXPECT(all.size() == 8); // eight distinct tensors, none shared

  for (Tensor *t : g)
    for (std::size_t n = 0; n < t->numel(); ++n)
      EXPECT(t->data()[n] == 0.0f);
  std::cout << "feedforward (4 parameters, 4 distinct gradients, shapes): ok\n";
}

void test_seeds() {
  FeedForward a(4, 8, 7), b(4, 8, 7), c(4, 8, 8);
  for (std::size_t i : {0, 2}) {
    EXPECT(same_data(*a.parameters()[i], *b.parameters()[i]));
    EXPECT(!same_data(*a.parameters()[i], *c.parameters()[i]));
  }
  // The two layers must not start from the same random stream: W1 and W2
  // share a shape up to transpose, so compare their raw values.
  const Tensor &w1 = *a.parameters()[0];
  const Tensor &w2 = *a.parameters()[2];
  EXPECT(w1.numel() == w2.numel());
  bool differ = false;
  for (std::size_t n = 0; n < w1.numel(); ++n)
    if (w1.data()[n] != w2.data()[n])
      differ = true;
  EXPECT(differ);
  std::cout << "feedforward (seeded init, layers get different seeds): ok\n";
}

// =====================================================================
// hand-checked example
// =====================================================================
// d_model 2, d_ff 3.  W1 = [[1,0],[0,1],[1,1]], b1 = [0,0,-5],
// W2 = [[1,0,0],[0,1,1]], b2 = [10,20].
//   X row 0 = [2,-1]: h_pre = [2,-1,-4] -> relu [2,0,0]  -> Y = [12,20]
//   X row 1 = [3, 4]: h_pre = [3, 4, 2] -> relu [3,4,2]  -> Y = [13,26]

void load_hand_weights(FeedForward &ff) {
  auto p = ff.parameters();
  set_values(*p[0], {1, 0, 0, 1, 1, 1});
  set_values(*p[1], {0, 0, -5});
  set_values(*p[2], {1, 0, 0, 0, 1, 1});
  set_values(*p[3], {10, 20});
}

void test_hand_forward() {
  FeedForward ff(2, 3, 0);
  load_hand_weights(ff);
  Tensor X({2, 2}), Y({2, 2});
  set_values(X, {2, -1, 3, 4});
  Y.fill(99.0f); // stale values must be overwritten
  ff.forward(X, Y);
  // std::cout << Y.data()[0] << " " << Y.data()[1] << " " << Y.data()[2] << " "
  //           << Y.data()[3];
  EXPECT(Y.data()[0] == 12 && Y.data()[1] == 20);
  EXPECT(Y.data()[2] == 13 && Y.data()[3] == 26);
  std::cout << "feedforward forward (hand-checked, stale output): ok\n";
}

// dY = ones. By hand:
//   dX      = [[1,0],[2,2]]
//   dW1     = [[5,3],[3,4],[3,4]]      db1 = [2,1,1]
//   dW2     = [[5,4,2],[5,4,2]]        db2 = [2,2]
void test_hand_backward() {
  FeedForward ff(2, 3, 0);
  load_hand_weights(ff);
  Tensor X({2, 2}), Y({2, 2}), dY({2, 2}), dX({2, 2});
  set_values(X, {2, -1, 3, 4});
  dY.fill(1.0f);

  ff.zero_grad();
  ff.forward(X, Y);
  ff.backward(X, dY, dX, false);

  const float edx[4] = {1, 0, 2, 2};
  for (std::size_t n = 0; n < 4; ++n)
    EXPECT(dX.data()[n] == edx[n]);

  auto g = ff.gradients();
  const std::vector<std::vector<float>> expected = {
      {5, 3, 3, 4, 3, 4}, {2, 1, 1}, {5, 4, 2, 5, 4, 2}, {2, 2}};
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT(g[i]->numel() == expected[i].size());
    for (std::size_t n = 0; n < expected[i].size(); ++n)
      EXPECT(g[i]->data()[n] == expected[i][n]);
  }
  std::cout << "feedforward backward (hand-checked, all gradients): ok\n";
}

// =====================================================================
// wiring against hand-composed layers
// =====================================================================

void test_matches_composed_layers() {
  std::mt19937 rng(1);
  const std::size_t d_model = 5, d_ff = 9, rows = 6;
  FeedForward ff(d_model, d_ff, 3);
  for (std::size_t i : {1, 3}) // non-zero biases
    fill_random(*ff.parameters()[i], rng);
  Reference ref(d_model, d_ff, ff);

  Tensor X({rows, d_model}), dY({rows, d_model});
  fill_random(X, rng);
  fill_random(dY, rng);

  // forward
  Tensor Y({rows, d_model}), Y_ref({rows, d_model});
  Tensor H1({rows, d_ff}), H2({rows, d_ff});
  ff.forward(X, Y);
  ref.l1.forward(X, H1);
  ref.relu.forward(H1, H2);
  ref.l2.forward(H2, Y_ref);
  EXPECT(same_data(Y, Y_ref, 1e-6f));

  // backward
  Tensor dX({rows, d_model}), dX_ref({rows, d_model});
  Tensor dH2({rows, d_ff}), dH1({rows, d_ff});
  ff.zero_grad();
  ff.backward(X, dY, dX, false);
  ref.l2.backward(H2, dY, dH2, false);
  ref.relu.backward(H1, H2, dH2, dH1, false);
  ref.l1.backward(X, dH1, dX_ref, false);
  EXPECT(same_data(dX, dX_ref, 1e-6f));

  auto g = ff.gradients();
  EXPECT(same_data(*g[0], *ref.l1.gradients()[0], 1e-5f));
  EXPECT(same_data(*g[1], *ref.l1.gradients()[1], 1e-5f));
  EXPECT(same_data(*g[2], *ref.l2.gradients()[0], 1e-5f));
  EXPECT(same_data(*g[3], *ref.l2.gradients()[1], 1e-5f));
  std::cout << "feedforward (matches hand-composed Linear-ReLU-Linear): ok\n";
}

void test_rank3_matches_flat() {
  std::mt19937 rng(2);
  const std::size_t d_model = 4, d_ff = 7;
  FeedForward ff(d_model, d_ff, 5);
  for (std::size_t i : {1, 3})
    fill_random(*ff.parameters()[i], rng);

  Tensor X3({2, 3, d_model}), X2({6, d_model});
  fill_random(X3, rng);
  for (std::size_t n = 0; n < X3.numel(); ++n)
    X2.data()[n] = X3.data()[n];
  Tensor dY3({2, 3, d_model}), dY2({6, d_model});
  fill_random(dY3, rng);
  for (std::size_t n = 0; n < dY3.numel(); ++n)
    dY2.data()[n] = dY3.data()[n];

  Tensor Y3({2, 3, d_model}), Y2({6, d_model});
  Tensor dX3({2, 3, d_model}), dX2({6, d_model});

  ff.forward(X3, Y3);
  ff.zero_grad();
  ff.backward(X3, dY3, dX3, false);
  std::vector<Tensor> grads3;
  for (Tensor *g : ff.gradients())
    grads3.push_back(g->clone());

  ff.forward(X2, Y2);
  ff.zero_grad();
  ff.backward(X2, dY2, dX2, false);

  for (std::size_t n = 0; n < Y3.numel(); ++n) {
    EXPECT(Y3.data()[n] == Y2.data()[n]);
    EXPECT(dX3.data()[n] == dX2.data()[n]);
  }
  auto g2 = ff.gradients();
  for (std::size_t i = 0; i < 4; ++i)
    EXPECT(same_data(grads3[i], *g2[i], 1e-6f));
  std::cout << "feedforward (rank 3 == flattened rank 2, outputs and "
               "gradients): ok\n";
}

// =====================================================================
// buffer lifecycle
// =====================================================================

void test_backward_requires_forward() {
  FeedForward ff(3, 5, 1);
  Tensor X({4, 3}), dY({4, 3}), dX({4, 3});
  EXPECT(throws<std::logic_error>([&] { ff.backward(X, dY, dX); }));

  // backward on a different shape than the last forward must also throw
  Tensor Y({4, 3});
  ff.forward(X, Y);
  Tensor X2({5, 3}), dY2({5, 3}), dX2({5, 3});
  EXPECT(throws<std::logic_error>([&] { ff.backward(X2, dY2, dX2); }));
  // and the matching call still works
  ff.backward(X, dY, dX);
  std::cout
      << "feedforward (backward before/after mismatched forward throws): ok\n";
}

void test_shape_changes() {
  std::mt19937 rng(3);
  FeedForward ff(3, 6, 2);
  Tensor X1({4, 3}), Y1({4, 3});
  fill_random(X1, rng);
  ff.forward(X1, Y1);
  Tensor Y1_again({4, 3});
  ff.forward(X1, Y1_again); // same shape twice: same result
  EXPECT(same_data(Y1, Y1_again));

  Tensor X2({2, 5, 3}), Y2({2, 5, 3}), dY2({2, 5, 3}), dX2({2, 5, 3});
  fill_random(X2, rng);
  fill_random(dY2, rng);
  ff.forward(X2, Y2); // different rank and size: buffers must be rebuilt
  ff.backward(X2, dY2, dX2);

  Tensor Y1_third({4, 3}); // back to the first shape: result unchanged
  ff.forward(X1, Y1_third);
  EXPECT(same_data(Y1, Y1_third));
  std::cout
      << "feedforward (buffers rebuilt on shape change, results stable): ok\n";
}

void test_misuse() {
  FeedForward ff(3, 5, 1);
  Tensor bad_d({4, 4}), Y({4, 4});
  EXPECT(throws<std::invalid_argument>([&] { ff.forward(bad_d, Y); }));
  Tensor X({4, 3}), wrong_Y({4, 5});
  EXPECT(throws<std::invalid_argument>([&] { ff.forward(X, wrong_Y); }));

  Tensor Yok({4, 3}), dY({4, 3}), bad_dX({2, 3});
  ff.forward(X, Yok);
  EXPECT(throws<std::invalid_argument>([&] { ff.backward(X, dY, bad_dX); }));
  std::cout << "feedforward (shape misuse throws invalid_argument): ok\n";
}

// =====================================================================
// accumulation
// =====================================================================

void test_accumulation() {
  std::mt19937 rng(4);
  const std::size_t d_model = 4, d_ff = 6, rows = 5;
  FeedForward ff(d_model, d_ff, 6);
  for (std::size_t i : {1, 3})
    fill_random(*ff.parameters()[i], rng);

  Tensor X({rows, d_model}), Y({rows, d_model}), dY({rows, d_model});
  Tensor dX({rows, d_model});
  fill_random(X, rng);
  fill_random(dY, rng);
  ff.forward(X, Y);

  ff.zero_grad();
  ff.backward(X, dY, dX, false);
  Tensor dX_once = dX.clone();
  std::vector<Tensor> once;
  for (Tensor *g : ff.gradients())
    once.push_back(g->clone());

  // Parameter gradients always accumulate across backward calls.
  ff.backward(X, dY, dX, false);
  auto g = ff.gradients();
  for (std::size_t i = 0; i < 4; ++i)
    for (std::size_t n = 0; n < g[i]->numel(); ++n)
      EXPECT(std::abs(g[i]->data()[n] - 2.0f * once[i].data()[n]) < 1e-5f);
  // dX with accumulate=false is overwritten, not doubled.
  EXPECT(same_data(dX, dX_once, 1e-6f));

  // dX with accumulate=true adds to what is there.
  dX.fill(10.0f);
  ff.backward(X, dY, dX, true);
  for (std::size_t n = 0; n < dX.numel(); ++n)
    EXPECT(std::abs(dX.data()[n] - (dX_once.data()[n] + 10.0f)) < 1e-5f);

  // zero_grad() through the base class clears every gradient of both layers.
  Module &base = ff;
  base.zero_grad();
  for (Tensor *t : base.gradients())
    for (std::size_t n = 0; n < t->numel(); ++n)
      EXPECT(t->data()[n] == 0.0f);
  std::cout << "feedforward (accumulation, zero_grad via Module&): ok\n";
}

// =====================================================================
// gradcheck
// =====================================================================

// ReLU has a kink at 0, and the numeric gradient is wrong if a perturbation
// crosses it. Search for a seed whose pre-activations all sit well away from 0
// (perturbing any input or parameter by eps moves a pre-activation by <= ~eps).
void test_gradcheck() {
  const std::size_t d_model = 4, d_ff = 8, rows = 3;
  const double eps = 1e-2, tol = 2e-2, margin = 0.05;

  std::unique_ptr<FeedForward> ff;
  Tensor X({rows, d_model}), Y({rows, d_model}), R({rows, d_model});
  Tensor dX({rows, d_model});

  bool found = false;
  for (unsigned seed = 1; seed <= 500 && !found; ++seed) {
    std::mt19937 rng(seed);
    ff = std::make_unique<FeedForward>(d_model, d_ff, seed);
    for (std::size_t i : {1, 3})
      fill_random(*ff->parameters()[i], rng);
    fill_random(X, rng);
    fill_random(R, rng);

    const Tensor &W1 = *ff->parameters()[0];
    const Tensor &b1 = *ff->parameters()[1];
    double min_abs = 1e9;
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t k = 0; k < d_ff; ++k) {
        double h = b1.data()[k];
        for (std::size_t j = 0; j < d_model; ++j)
          h += static_cast<double>(X.data()[r * d_model + j]) *
               W1.data()[k * d_model + j];
        min_abs = std::min(min_abs, std::abs(h));
      }
    found = min_abs > margin;
  }
  EXPECT(found); // some seed keeps every pre-activation away from the kink

  auto loss = [&]() {
    ff->forward(X, Y);
    return dot_loss(Y, R);
  };

  loss();
  ff->zero_grad();
  ff->backward(X, /*dY=*/R, dX, false);

  auto params = ff->parameters();
  auto grads = ff->gradients();
  const char *names[4] = {"W1", "b1", "W2", "b2"};

  std::cout << "feedforward gradcheck:\n";
  GradcheckResult rx = functional::check_gradient(X, dX, loss, eps, tol);
  std::cout << "  X : max rel err " << rx.max_rel_error << "\n";
  EXPECT(rx.passed);
  for (std::size_t i = 0; i < 4; ++i) {
    GradcheckResult r =
        functional::check_gradient(*params[i], *grads[i], loss, eps, tol);
    std::cout << "  " << names[i] << ": max rel err " << r.max_rel_error
              << "\n";
    EXPECT(r.passed);
  }

  // Self-tests: a wrong gradient must FAIL, for the first layer and the second.
  for (std::size_t i : {0, 2}) {
    Tensor bad = grads[i]->clone();
    for (std::size_t n = 0; n < bad.numel(); ++n)
      bad.data()[n] *= 1.1f;
    EXPECT(!functional::check_gradient(*params[i], bad, loss, eps, tol).passed);
  }
  // Self-test: the parameters returned AS gradients (the earlier wiring slip)
  // must FAIL.
  EXPECT(!functional::check_gradient(*params[2], *params[2], loss, eps, tol)
              .passed);
  std::cout << "feedforward gradcheck (incl. self-tests): ok\n";
}

} // namespace

namespace tests {
void run_feedforward() {
  test_structure();
  test_seeds();

  test_hand_forward();
  test_hand_backward();
  test_matches_composed_layers();
  test_rank3_matches_flat();

  test_backward_requires_forward();
  test_shape_changes();
  test_misuse();

  test_accumulation();
  test_gradcheck();

  std::cout << "all feedforward tests passed\n";
}
} // namespace tests
