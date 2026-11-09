#include <cassert>
#include <cmath>
#include <core/errors.h>
#include <core/random.h>
#include <core/tensor.h>
#include <nn/initialization.h>
#include <optional>
#include <random>
#include <stdexcept>

namespace crandom = core::random;
using core::Tensor;

namespace error = core::error;

namespace nn::initialize {

template <typename Sample> void fill_dist(Tensor &t, Sample &&sample) {
  float *_t = t.data();

  for (std::size_t i = 0; i < t.numel(); ++i) {
    _t[i] = sample();
  }
}

Fans get_fans(const Tensor &t) {
  if (t.rank() < 2) {
    throw error::ShapeError(
        "utils::get_fans",
        error::make_message(
            "fan based init needs Tensor rank >= 2, current rank: ", t.rank()),
        t.shape());
  }

  std::size_t receptive = 1;
  for (std::size_t d = 2; d < t.rank(); ++d) {
    receptive *= t.shape()[d];
  }

  const std::size_t out = t.shape()[0];
  const std::size_t in = t.shape()[1];

  if (out == 0 || in == 0) {
    throw error::ShapeError("utils::get_fans", "contains zero-size dimensions",
                            t.shape());
    throw std::invalid_argument("fan based init: zero-size dimensions");
  }

  return {static_cast<float>(in * receptive),
          static_cast<float>(out * receptive)};
}

void fill_random(Tensor &t, const Initialize &init, crandom::Generator &gen) {
  switch (init.kind) {
  case InitializeKind::Zeros: {
    t.fill(0.0f);
    break;
  }
  case InitializeKind::Constant: {
    t.fill(init.a);
    break;
  }
  case InitializeKind::Uniform: {
    assert(init.a <= init.b);
    fill_dist(t, [&]() { return gen.uniform(init.a, init.b); });
    break;
  }
  case InitializeKind::Normal: {
    assert(init.b >= 0.0f);
    fill_dist(t, [&]() { return gen.normal(init.a, init.b); });
    break;
  }
  case InitializeKind::XavierUniform: {
    Fans f = get_fans(t);
    const float bound = std::sqrt(6.0f / (f.fan_in + f.fan_out));
    fill_dist(t, [&]() { return gen.uniform(-bound, bound); });
    break;
  }
  case InitializeKind::XavierNormal: {
    Fans f = get_fans(t);
    const float stddev = std::sqrt(2.0f / (f.fan_in + f.fan_out));
    fill_dist(t, [&]() { return gen.normal(0.0f, stddev); });
    break;
  }
  case InitializeKind::KaimingUniform: {
    Fans f = get_fans(t);
    const float bound = std::sqrt(6.0f / f.fan_in);
    fill_dist(t, [&]() { return gen.uniform(-bound, bound); });
    break;
  }
  case InitializeKind::KaimingNormal: {
    Fans f = get_fans(t);
    const float stddev = std::sqrt(2.0f / f.fan_in);
    fill_dist(t, [&]() { return gen.normal(0.0f, stddev); });
    break;
  }
  }
}

void fill_random(Tensor &t, const Initialize &init) {
  fill_random(t, init, crandom::global_rng());
}

} // namespace nn::initialize
