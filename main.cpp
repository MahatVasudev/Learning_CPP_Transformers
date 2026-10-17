#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <tests/tests.h>

namespace {

struct Suite {
  const char *name;
  void (*run)();
};

const Suite kSuites[] = {
    {"activations", tests::run_activations},
    {"cross_entropy", tests::run_cross_entropy},
    {"embedding", tests::run_embedding},
    {"layernorm", tests::run_layernorm},
    {"linear", tests::run_linear},
    {"matmul", tests::run_matmul},
    {"tensor", tests::run_tensor},
    {"utils", tests::run_utils},
    {"positional_encoding", tests::run_positional_encoding},
    {"feed_forward", tests::run_feedforward},
    {"lazy_gradient", tests::run_module_contract}};

// Optional filter: `--test embedding` runs only suites whose name contains it.
int run_tests(const char *filter) {
  int ran = 0, failed = 0;
  for (const Suite &s : kSuites) {
    if (filter && std::string(s.name).find(filter) == std::string::npos)
      continue;
    ++ran;
    std::cout << "=== " << s.name << " ===\n";
    try {
      s.run();
    } catch (const std::exception &e) {
      std::cerr << "SUITE FAILED: " << s.name << ": " << e.what() << "\n";
      ++failed;
    }
  }
  std::cout << "\n" << (ran - failed) << "/" << ran << " suites passed\n";
  return (ran > 0 && failed == 0) ? 0 : 1; // a filter matching nothing fails
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 1 && std::strcmp(argv[1], "--test") == 0)
    return run_tests(argc > 2 ? argv[2] : nullptr);

  // ... your normal program goes here ...
  return 0;
}
