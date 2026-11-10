#include <core/errors.h>
#include <core/tensor.h>
#include <core/utils.h>
#include <nn/initialization.h>
#include <nn/module.h>
#include <ops/matmul.h>

#include <vector>

using core::Tensor;
namespace error = core::error;

namespace nn {

std::vector<Tensor *>
Module::concat_parameters(const std::vector<Module *> &modules) {
  std::vector<Tensor *> global_vec;
  for (Module *mod : modules) {
    assert(mod != nullptr);
    const std::vector<Tensor *> p = mod->parameters();
    global_vec.insert(global_vec.end(), p.begin(), p.end());
  }

  return global_vec;
}

std::vector<Tensor *>
Module::concat_gradients(const std::vector<Module *> &modules) {
  std::vector<Tensor *> global_vec;
  for (Module *mod : modules) {
    assert(mod != nullptr);
    const std::vector<Tensor *> g = mod->gradients();
    global_vec.insert(global_vec.end(), g.begin(), g.end());
  }

  return global_vec;
}

void Module::set_training_all(const std::vector<Module *> &modules, bool mode) {
  for (Module *mod : modules) {
    assert(mod != nullptr);
    mod->set_training(mode);
  }
}

void Module::clear_cache_all(const std::vector<Module *> &modules) {
  for (Module *mod : modules) {
    assert(mod != nullptr);
    mod->clear_cache();
  }
}

void Module::require_training(std::string method_name,
                              std::string additional_message) {
  if (!training_)
    throw error::StateError(
        method_name +
        ": cannot run this method on eval(), run train() first\n" +
        additional_message);
}

} // namespace nn
