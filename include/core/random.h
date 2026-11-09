#ifndef RANDOM_NN_H

#define RANDOM_NN_H

#include <random>

namespace core::random {

class Generator {

public:
  Generator(unsigned seed = 0) : engine_(seed), normal_dist_(0.0f, 1.0f) {}
  void set_seed(unsigned seed) {
    engine_.seed(seed);
    normal_dist_.reset();
  }

  float uniform() { return uniform_dist_(engine_); }
  float uniform(float low, float high) {
    return low + (high - low) * uniform_dist_(engine_);
  }

  float normal() { return normal_dist_(engine_); }

  float normal(float mean, float stddev) {
    return mean + stddev * normal_dist_(engine_);
  }

private:
  std::mt19937 engine_;
  std::uniform_real_distribution<float> uniform_dist_;
  std::normal_distribution<float> normal_dist_;
};

Generator &global_rng();

void set_rng(std::uint64_t seed);

} // namespace core::random

#endif // !RANDOM_NN_H
