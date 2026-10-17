#ifndef UTILS_INIT_H

#define UTILS_INIT_H

#include <functional/tensor.h>
#include <vector>

namespace utils {

enum class InitializeKind {
  Zeros,
  Constant,
  Uniform,
  Normal,
  XavierUniform,
  XavierNormal,
  KaimingUniform,
  KaimingNormal
};

struct Fans {
  float fan_in;
  float fan_out;
};

Fans get_fans(const functional::Tensor &t);

struct Initialize {
  InitializeKind kind = InitializeKind::XavierUniform;

  float a = 0.0f;
  float b = 1.0f;

  static Initialize zeros() { return {InitializeKind::Zeros, 0.0f, 0.0f}; }
  static Initialize constant(const float v) {
    return {InitializeKind::Constant, v, 0.0f};
  }
  static Initialize uniform(float low, float high) {
    return {InitializeKind::Uniform, low, high};
  }
  static Initialize normal(float mean, float stddev) {
    return {InitializeKind::Normal, mean, stddev};
  }
  static Initialize xavier_uniform() {
    return {InitializeKind::XavierUniform, 0.0f, 0.0f};
  }
  static Initialize xavier_normal() {
    return {InitializeKind::XavierNormal, 0.0f, 0.0f};
  }

  static Initialize kaiming_uniform() {
    return {InitializeKind::KaimingUniform, 0.0f, 0.0f};
  }
  static Initialize kaiming_normal() {
    return {InitializeKind::KaimingNormal, 0.0f, 0.0f};
  }
};

void fill_random(functional::Tensor &t, const Initialize &init,
                 const unsigned seed);

template <typename T> T iterative_product_vector(std::vector<T> &data);

} // namespace utils
  //
#endif // !UTILS_INIT_H
