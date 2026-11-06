#ifndef INITIALIZATION_NN_H

#define INITIALIZATION_NN_H

#include <core/tensor.h>

namespace nn::initialize {

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

Fans get_fans(const core::Tensor &t);

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

void fill_random(core::Tensor &t, const Initialize &init, const unsigned seed);

} // namespace nn::initialize
  //
#endif // !UTILS_INIT_H
