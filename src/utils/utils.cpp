#include <cassert>
#include <cmath>
#include <functional/tensor.h>
#include <random>
#include <stdexcept>
#include <utils/utils.h>

namespace utils {

template <typename Dist>
void fill_dist(functional::Tensor &t, Dist dist, std::mt19937 rng) {

  float *_t = t.data();

  for (std::size_t i = 0; i < t.numel(); ++i) {
    _t[i] = dist(rng);
  }
}

Fans get_fans(const functional::Tensor &t) {
  if (t.rank() < 2) {
    throw std::invalid_argument("fan based init needs Tensor rank >= 2");
  }

  std::size_t receptive = 1;
  for (std::size_t d = 2; d < t.rank(); ++d) {
    receptive *= t.shape()[d];
  }

  const std::size_t out = t.shape()[0];
  const std::size_t in = t.shape()[1];

  if (out == 0 || in == 0) {
    throw std::invalid_argument("fan based init: zero-size dimensions");
  }

  return {static_cast<float>(in * receptive),
          static_cast<float>(out * receptive)};
}

void fill_random(functional::Tensor &t, const Initialize &init,
                 const unsigned seed) {
  std::mt19937 rng(seed);
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
    fill_dist(t, std::uniform_real_distribution<float>(init.a, init.b), rng);
    break;
  }
  case InitializeKind::Normal: {
    assert(init.b >= 0.0f);
    fill_dist(t, std::normal_distribution<float>(init.a, init.b), rng);
    break;
  }
  case InitializeKind::XavierUniform: {
    Fans f = get_fans(t);
    const float bound = std::sqrt(6.0f / (f.fan_in + f.fan_out));
    fill_dist(t, std::uniform_real_distribution<float>(-bound, bound), rng);
    break;
  }
  case InitializeKind::XavierNormal: {
    Fans f = get_fans(t);
    const float stddev = std::sqrt(2.0f / (f.fan_in + f.fan_out));
    fill_dist(t, std::normal_distribution<float>(0.0f, stddev), rng);
    break;
  }
  case InitializeKind::KaimingUniform: {
    Fans f = get_fans(t);
    const float bound = std::sqrt(6.0f / f.fan_in);
    fill_dist(t, std::uniform_real_distribution<float>(-bound, bound), rng);
    break;
  }
  case InitializeKind::KaimingNormal: {
    Fans f = get_fans(t);
    const float stddev = std::sqrt(2.0f / f.fan_in);
    fill_dist(t, std::normal_distribution<float>(0.0f, stddev), rng);
    break;
  }
  }
}

} // namespace utils
