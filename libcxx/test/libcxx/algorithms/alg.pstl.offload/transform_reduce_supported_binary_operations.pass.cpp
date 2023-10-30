//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test will fail if the number of devices detected by OpenMP is larger
// than zero but std::transform_reduce(std::execution::par_unseq,...) is not
// executed on the device.

// UNSUPPORTED: c++03, c++11, c++14, gcc

// ADDITIONAL_COMPILE_FLAGS: -O2 -Wno-pass-failed -fopenmp

// REQUIRES: openmp_pstl_backend

#include <algorithm>
#include <cassert>
#include <execution>
#include <functional>
#include <vector>
#include <omp.h>

int main(void) {
  // We only run the test if a device is detected by OpenMP
  if (omp_get_num_devices() < 1)
    return 0;

  // Initializing test array
  const int test_size = 10000;

  // Addition with doubles
  {
    std::vector<double> v(test_size, 1.0);
    std::vector<double> w(test_size, 2.0);
    double result = std::transform_reduce(
        std::execution::par_unseq, v.begin(), v.end(), w.begin(), 5.0, std::plus{}, [](double& a, double& b) {
          return 0.5 * (b - a) * ((double)!omp_is_initial_device());
        });
    assert((std::abs(result - 0.75 * ((double)test_size - 5.0)) < 1e-8) &&
           "std::transform_reduce(std::execution::par_unseq,...) does not have the intended effect for the binary "
           "operation std::plus.");
  }

  // Subtraction of floats
  {
    std::vector<float> v(test_size, 1.0f);
    std::vector<float> w(test_size, 1.5f);
    double result = std::transform_reduce(
        std::execution::par_unseq,
        v.begin(),
        v.end(),
        w.begin(),
        1.25 * ((float)test_size),
        std::minus{},
        [](float& a, float& b) { return 0.5 * (a + b) * ((float)!omp_is_initial_device()); });
    assert((std::abs(result) < 1e-8f) && "std::transform_reduce(std::execution::par_unseq,...) does not have the "
                                         "intended effect for the binary operation std::minus.");
  }
  return 0;
}