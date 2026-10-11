#include "functional/tensor.h"

#include "tests/tests.h"
#include <cassert>
#include <iostream>
#include <utility>

using namespace tests;
using functional::Tensor;

static Tensor make() {
  Tensor t({2, 3});
  t.fill(7.0f);
  return t; // moved (or elided), never deep-copied
}

void tests::run_tensor() {
  // 1. Rank-3 indexing: strides for (2,3,4) must be (12,4,1).
  Tensor a({2, 3, 4});
  assert(a.numel() == 24);
  assert(a.strides()[0] == 12 && a.strides()[1] == 4 && a.strides()[2] == 1);
  a(1, 2, 3) = 5.0f;
  assert(a.data()[1 * 12 + 2 * 4 + 3] == 5.0f);

  // 2. Copy is deep: changing the copy must not change the original.
  Tensor b = a;
  b(1, 2, 3) = 9.0f;
  assert(a(1, 2, 3) == 5.0f);

  // 3. Copy assignment, including self-assignment.
  Tensor c({1});
  c = a;
  assert(c(1, 2, 3) == 5.0f);
  Tensor &cr = c;
  c = cr; // self-assign: must not crash or corrupt
  assert(c(1, 2, 3) == 5.0f);

  // 4. Move leaves the source empty.
  Tensor d = std::move(a);
  assert(d(1, 2, 3) == 5.0f);
  assert(a.numel() == 0 && a.data() == nullptr);

  // 5. Returning from a function.
  Tensor e = make();
  assert(e(1, 2) == 7.0f);

  // 6. Move assignment.
  Tensor f({2});
  f = std::move(e);
  assert(f(1, 2) == 7.0f && e.numel() == 0);

  std::cout << "all tests passed\n";
}
