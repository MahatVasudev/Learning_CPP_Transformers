#ifndef DECODER_TRANSFORMER_NN

#define DECODER_TRANSFORMER_NN

#include "nn/module.h"
namespace nn::transformers {

class DecoderUnit final : public Module {

public:
  std::vector<Tensor *> parameters() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `parameters` is not implemented.");
  }

  std::vector<Tensor *> gradients() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `gradients` is not implemented.");
  }

private:
};

class DecoderLayer;

} // namespace nn::transformers

#endif // !DECODER_TRANSFORMER_NN
