#include <errors/errors.h>
#include <sstream>
#include <string>

namespace error {

std::string format_shape(const std::vector<std::size_t> &shape) {
  std::ostringstream oss;
  oss << '(';

  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << shape[i];
  }

  oss << ')';

  return oss.str();
}

ShapeError::ShapeError(std::string operation, std::string expectation,
                       const std::vector<std::size_t> found_shape,
                       std::string additional_string)
    : std::invalid_argument(build_message_(operation, expectation, found_shape,
                                           additional_string)) {}

std::string
ShapeError::build_message_(const std::string &operation,
                           const std::string &expectation,
                           const std::vector<std::size_t> &found_shape,
                           const std::string &additional_string) {

  return make_message("ShapeError: ", operation, ": expected ", expectation,
                      ", got shape ", format_shape(found_shape), '\n',
                      additional_string);
}

IndexError::IndexError(std::string message, const std::size_t found_index,
                       std::string additional_message)
    : std::out_of_range(
          build_message_(message, found_index, additional_message)) {}

std::string IndexError::build_message_(const std::string &message,
                                       const std::size_t &found_index,
                                       const std::string &additional_message) {
  return make_message("IndexError: ", message, ": found index ", found_index,
                      '\n', additional_message);
}

StateError::StateError(const std::string &message)
    : std::logic_error("StateError: " + message) {}

} // namespace error
