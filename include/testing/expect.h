#ifndef TEST_EXPECT_H

#include <sstream>
#include <stdexcept>
#define TEST_EXPECT_H

#pragma once
// Throws instead of exiting, so the runner can report which suite failed.
#define EXPECT(cond)                                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::ostringstream os_;                                                  \
      os_ << "EXPECT failed: " #cond " (" << __FILE__ << ":" << __LINE__       \
          << ")";                                                              \
      throw std::runtime_error(os_.str());                                     \
    }                                                                          \
  } while (0)

#endif // !TEST_EXPECT_H
