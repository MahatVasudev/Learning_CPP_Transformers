#ifndef TEST_TEST_BASIC_H
#define TEST_TEST_BASIC_H

#pragma once

namespace tests {
void run_activations();
void run_cross_entropy();
void run_embedding();
void run_layernorm();
void run_linear();
void run_matmul();
void run_tensor();
void run_utils();
void run_positional_encoding();
void run_feedforward();
void run_module_contract();
} // namespace tests

#endif // !TEST_TEST_BASIC_H
