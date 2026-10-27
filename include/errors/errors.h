
#ifndef ERROR_TENSOR_H

#define ERROR_TENSOR_H

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace error {

std::string format_shape(const std::vector<std::size_t> &shape);

template <typename... Args> std::string make_message(const Args &...args) {
  std::ostringstream oss;
  (oss << ... << args);

  return oss.str();
}
class ShapeError : public std::invalid_argument {

public:
  explicit ShapeError(std::string operation, std::string expectation,
                      const std::vector<std::size_t> found_shape,
                      std::string additional_message = "");

private:
  std::string build_message_(const std::string &operation,
                             const std::string &expectation,
                             const std::vector<std::size_t> &found_shape,
                             const std::string &addtional_message);
};

class IndexError : public std::out_of_range {
public:
  template <typename T>
  explicit IndexError(std::string message, T found_index,
                      std::string additional_message = "")
      : std::out_of_range(
            build_message_(message, found_index, additional_message)) {}

private:
  template <typename T>
  std::string build_message_(const std::string &message, const T &found_index,
                             const std::string &additional_message) {
    return make_message("IndexError: ", message, ": found index ", found_index,
                        '\n', additional_message);
  }
};

class StateError : public std::logic_error {
public:
  explicit StateError(const std::string &message);
};

} // namespace error

#endif // !ERROR_TENSOR_H
