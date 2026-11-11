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
    if (mod == nullptr)
      continue;
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

void Module::require_training(std::string method_name,
                              std::string additional_message) {
  if (!training_)
    throw error::StateError(
        method_name +
        ": cannot run this method on eval(), run train() first\n" +
        additional_message);
}

void Module::clear_cache() {
  this->clear_local_cache();
  for (Module *child : modules()) {
    if (child)
      child->clear_cache();
  }
}

std::vector<Tensor *>
Module::collect_parameters(std::vector<Tensor *> parameters) {
  std::vector<Tensor *> children = concat_parameters(modules());

  parameters.insert(parameters.end(), children.begin(), children.end());

  return parameters;
}

std::vector<Tensor *>
Module::collect_gradients(std::vector<Tensor *> gradients) {
  std::vector<Tensor *> children = concat_gradients(modules());

  gradients.insert(gradients.end(), children.begin(), children.end());

  return gradients;
}

} // namespace nn
