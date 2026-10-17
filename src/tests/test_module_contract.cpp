#include <functional/module.h> // adjust: wherever Module/Linear/LayerNorm/Embedding live
#include <functional/tensor.h>
#include <model/model.h>
#include <tests/expect.h>
#include <tests/tests.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using functional::Embedding;
using functional::LayerNorm;
using functional::Linear;
using functional::Module;
using functional::Tensor;
using model::FeedForward;

// ---------- helpers ----------

void fill_random(Tensor &t, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t n = 0; n < t.numel(); ++n)
    t.data()[n] = dist(rng);
}

template <typename Ex, typename F> bool throws(F f) {
  try {
    f();
  } catch (const Ex &) {
    return true;
  }
  return false;
}

// Copies of every gradient tensor, in gradients() order.
std::vector<Tensor> snapshot(Module &m) {
  std::vector<Tensor> s;
  for (Tensor *g : m.gradients())
    s.push_back(g->clone());
  return s;
}

bool any_nonzero(const std::vector<Tensor> &s) {
  for (const Tensor &t : s)
    for (std::size_t n = 0; n < t.numel(); ++n)
      if (t.data()[n] != 0.0f)
        return true;
  return false;
}

bool all_zero(Module &m) {
  for (Tensor *g : m.gradients())
    for (std::size_t n = 0; n < g->numel(); ++n)
      if (g->data()[n] != 0.0f)
        return false;
  return true;
}

// a == scale * b, elementwise, with a relative tolerance.
bool matches(const std::vector<Tensor> &a, const std::vector<Tensor> &b,
             float scale, float tol) {
  if (a.size() != b.size())
    return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].shape() != b[i].shape())
      return false;
    for (std::size_t n = 0; n < a[i].numel(); ++n) {
      const float expect = scale * b[i].data()[n];
      if (std::abs(a[i].data()[n] - expect) >
          tol * (1.0f + std::abs(b[i].data()[n])))
        return false;
    }
  }
  return true;
}

// ---------- one description per layer under test ----------

struct Case {
  std::string name;
  std::size_t n_tensors; // how many parameter / gradient tensors it owns
  std::function<std::unique_ptr<Module>()> make;
  std::function<void(Module &)> infer; // forward only
  std::function<void(Module &)> step;  // forward + backward (accumulates)
};

Case linear_case(bool bias) {
  const std::size_t in = 4, out = 5, rows = 6;
  Tensor X({rows, in}), dY({rows, out});
  fill_random(X, 1);
  fill_random(dY, 2);
  Case c;
  c.name = bias ? "linear (bias)" : "linear (no bias)";
  c.n_tensors = bias ? 2 : 1;
  c.make = [=] { return std::make_unique<Linear>(in, out, bias, 7u); };
  c.infer = [=](Module &m) {
    Tensor Y({rows, out});
    static_cast<Linear &>(m).forward(X, Y);
  };
  c.step = [=](Module &m) {
    auto &l = static_cast<Linear &>(m);
    Tensor Y({rows, out}), dX({rows, in});
    l.forward(X, Y);
    l.backward(X, dY, dX, false);
  };
  return c;
}

Case layernorm_case() {
  const std::size_t D = 6, rows = 5;
  Tensor X({rows, D}), dY({rows, D});
  fill_random(X, 3);
  fill_random(dY, 4);
  Case c;
  c.name = "layernorm";
  c.n_tensors = 2;
  c.make = [=] { return std::make_unique<LayerNorm>(D); };
  c.infer = [=](Module &m) {
    Tensor Y({rows, D});
    static_cast<LayerNorm &>(m).forward(X, Y);
  };
  c.step = [=](Module &m) {
    auto &l = static_cast<LayerNorm &>(m);
    Tensor Y({rows, D}), dX({rows, D});
    l.forward(X, Y);
    l.backward(X, dY, dX, false);
  };
  return c;
}

Case embedding_case() {
  const std::size_t vocab = 6, d = 4;
  const std::vector<std::int32_t> ids = {3, 1, 3, 0, 2, 1};
  const std::size_t n = ids.size();
  Tensor dY({n, d});
  fill_random(dY, 5);
  Case c;
  c.name = "embedding";
  c.n_tensors = 1;
  c.make = [=] { return std::make_unique<Embedding>(vocab, d, 7u); };
  c.infer = [=](Module &m) {
    Tensor Y({n, d});
    static_cast<Embedding &>(m).forward(ids.data(), n, Y);
  };
  c.step = [=](Module &m) {
    auto &e = static_cast<Embedding &>(m);
    Tensor Y({n, d});
    e.forward(ids.data(), n, Y);
    e.backward(ids.data(), n, dY);
  };
  return c;
}

Case feedforward_case() {
  const std::size_t d_model = 4, d_ff = 8, rows = 5;
  Tensor X({rows, d_model}), dY({rows, d_model});
  fill_random(X, 6);
  fill_random(dY, 7);
  Case c;
  c.name = "feedforward";
  c.n_tensors = 4;
  c.make = [=] { return std::make_unique<FeedForward>(d_model, d_ff, 7u); };
  c.infer = [=](Module &m) {
    Tensor Y({rows, d_model});
    static_cast<FeedForward &>(m).forward(X, Y);
  };
  c.step = [=](Module &m) {
    auto &f = static_cast<FeedForward &>(m);
    Tensor Y({rows, d_model}), dX({rows, d_model});
    f.forward(X, Y);
    f.backward(X, dY, dX, false);
  };
  return c;
}

// =====================================================================
// the contract, checked for every layer
// =====================================================================

// gradients() on a fresh layer allocates on demand: valid, zero-filled,
// never null, one per parameter, same shapes, nothing shared.
void test_fresh_gradients(const Case &c) {
  auto m = c.make();
  auto p = m->parameters();
  auto g = m->gradients();
  EXPECT(p.size() == c.n_tensors);
  EXPECT(g.size() == c.n_tensors);

  std::set<const Tensor *> seen(p.begin(), p.end());
  EXPECT(seen.size() == p.size()); // no parameter listed twice
  for (std::size_t i = 0; i < g.size(); ++i) {
    EXPECT(g[i] != nullptr);
    EXPECT(g[i]->shape() == p[i]->shape());
    EXPECT(
        seen.insert(g[i]).second); // distinct from every parameter and gradient
    for (std::size_t n = 0; n < g[i]->numel(); ++n)
      EXPECT(g[i]->data()[n] == 0.0f);
  }
}

// Asking twice must not reallocate or reset anything.
void test_stable_addresses(const Case &c) {
  auto m = c.make();
  auto g1 = m->gradients();
  g1[0]->data()[0] = 7.0f;
  auto g2 = m->gradients();
  EXPECT(g1.size() == g2.size());
  for (std::size_t i = 0; i < g1.size(); ++i)
    EXPECT(g1[i] == g2[i]);
  EXPECT(g2[0]->data()[0] == 7.0f);
}

// The `emplace`-on-every-backward bug: a second backward must ADD to the
// first one, not replace it.
void test_backward_accumulates(const Case &c) {
  auto m = c.make();
  m->zero_grad();
  c.step(*m);
  std::vector<Tensor> once = snapshot(*m);
  EXPECT(any_nonzero(once));
  c.step(*m);
  EXPECT(matches(snapshot(*m), once, 2.0f, 1e-5f));
}

void test_zero_grad_via_base(const Case &c) {
  auto m = c.make();
  c.step(*m);
  EXPECT(any_nonzero(snapshot(*m)));
  Module &base = *m;
  base.zero_grad();
  EXPECT(all_zero(base));
}

// train() / eval() flip the flag; forward is allowed in eval mode, backward
// is not (logic_error), and train() makes it work again.
void test_mode_and_backward_guard(const Case &c) {
  auto m = c.make();
  EXPECT(m->training());

  m->eval();
  EXPECT(!m->training());
  c.infer(*m); // inference must work
  EXPECT(throws<std::logic_error>([&] { c.step(*m); }));

  m->train();
  EXPECT(m->training());
  c.step(*m);
  EXPECT(any_nonzero(snapshot(*m)));
}

// eval() releases gradients (children too, for composites): after
// eval() -> train() they must come back zero-filled, and a new backward must
// give ONE step's gradient, not stale-plus-new.
// This also proves composites forward BOTH eval() and train() to children:
// a missed eval() leaves stale gradients; a missed train() makes step() throw.
void test_eval_releases_gradients(const Case &c) {
  auto m = c.make();
  m->zero_grad();
  c.step(*m);
  std::vector<Tensor> once = snapshot(*m);
  EXPECT(any_nonzero(once));

  m->eval();
  m->train();
  EXPECT(all_zero(*m));

  c.step(*m);
  EXPECT(matches(snapshot(*m), once, 1.0f, 1e-6f));
}

// Current policy: gradients() allocates on demand even in eval mode.
// If you later decide it should throw there, change (or delete) this test.
void test_gradients_in_eval_mode(const Case &c) {
  auto m = c.make();
  m->eval();
  auto g = m->gradients();
  EXPECT(g.size() == c.n_tensors);
  for (Tensor *t : g) {
    EXPECT(t != nullptr);
    for (std::size_t n = 0; n < t->numel(); ++n)
      EXPECT(t->data()[n] == 0.0f);
  }
}

// =====================================================================
// layer-specific checks
// =====================================================================

// gamma and beta have the same shape, so a swapped order passes every shape
// test. Check the VALUES: gradients()[1] is column sums of dY, [0] is
// sum(dY * xhat).
void test_layernorm_gradient_order() {
  const std::size_t D = 6, rows = 5;
  LayerNorm ln(D);
  Tensor X({rows, D}), Y({rows, D}), dY({rows, D}), dX({rows, D});
  fill_random(X, 11);
  fill_random(dY, 12);

  EXPECT(ln.parameters()[0] == &ln.gamma());
  EXPECT(ln.parameters()[1] == &ln.beta());

  ln.forward(X, Y);
  ln.zero_grad();
  ln.backward(X, dY, dX, false);
  auto g = ln.gradients();

  std::vector<double> dgamma(D, 0.0), dbeta(D, 0.0);
  for (std::size_t r = 0; r < rows; ++r) {
    double mean = 0.0, var = 0.0;
    for (std::size_t j = 0; j < D; ++j)
      mean += X.data()[r * D + j];
    mean /= static_cast<double>(D);
    for (std::size_t j = 0; j < D; ++j) {
      const double cdiff = X.data()[r * D + j] - mean;
      var += cdiff * cdiff;
    }
    var /= static_cast<double>(D);
    const double rstd = 1.0 / std::sqrt(var + 1e-5);
    for (std::size_t j = 0; j < D; ++j) {
      const double xhat = (X.data()[r * D + j] - mean) * rstd;
      dgamma[j] += static_cast<double>(dY.data()[r * D + j]) * xhat;
      dbeta[j] += dY.data()[r * D + j];
    }
  }
  for (std::size_t j = 0; j < D; ++j) {
    EXPECT(std::abs(g[0]->data()[j] - dgamma[j]) < 1e-4);
    EXPECT(std::abs(g[1]->data()[j] - dbeta[j]) < 1e-4);
  }
  EXPECT(std::abs(dgamma[0] - dbeta[0]) >
         1e-3); // the two differ, so the check means something
}

// With bias off, the bias gradient must neither exist in the list nor
// pair up wrongly with the parameters.
void test_linear_bias_free_lists() {
  Linear no_bias(4, 5, /*bias=*/false, 3);
  EXPECT(no_bias.parameters().size() == 1);
  EXPECT(no_bias.gradients().size() == 1);
  EXPECT(no_bias.gradients()[0]->shape() == no_bias.parameters()[0]->shape());
  Linear with_bias(4, 5, /*bias=*/true, 3);
  EXPECT(with_bias.parameters().size() == 2);
  EXPECT(with_bias.gradients().size() == 2);
}

// Compiles only if set_training is public on every concrete type.
void test_set_training_is_public() {
  Linear l(2, 3);
  l.set_training(false);
  EXPECT(!l.training());
  LayerNorm ln(3);
  ln.set_training(false);
  EXPECT(!ln.training());
  Embedding e(4, 2);
  e.set_training(false);
  EXPECT(!e.training());
  FeedForward ff(2, 4);
  ff.set_training(false);
  EXPECT(!ff.training());
  ff.set_training(true);
  EXPECT(ff.training());
}

} // namespace

namespace tests {
void run_module_contract() {
  std::vector<Case> cases;
  cases.push_back(linear_case(true));
  cases.push_back(linear_case(false));
  cases.push_back(layernorm_case());
  cases.push_back(embedding_case());
  cases.push_back(feedforward_case());

  for (const Case &c : cases) {
    test_fresh_gradients(c);
    test_stable_addresses(c);
    test_backward_accumulates(c);
    test_zero_grad_via_base(c);
    test_mode_and_backward_guard(c);
    test_eval_releases_gradients(c);
    test_gradients_in_eval_mode(c);
    std::cout << "module contract (" << c.name << "): ok\n";
  }

  test_layernorm_gradient_order();
  std::cout << "layernorm (gradient order: gamma then beta, by value): ok\n";
  test_linear_bias_free_lists();
  std::cout << "linear (bias-free lists have one entry each): ok\n";
  test_set_training_is_public();
  std::cout << "set_training is public on every layer: ok\n";

  std::cout << "all module contract tests passed\n";
}
} // namespace tests
