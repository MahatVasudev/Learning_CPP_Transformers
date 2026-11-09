#include <core/random.h>

namespace core::random {
Generator &global_rng() {
  static Generator generator(12345);
  return generator;
};
void set_rng(std::uint64_t seed) { global_rng().set_seed(seed); };
} // namespace core::random
