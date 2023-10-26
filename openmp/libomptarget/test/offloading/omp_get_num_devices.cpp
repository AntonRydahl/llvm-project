// RUN: %libomptarget-compilexx-run-and-check-generic

#include <omp.h>
#include <iostream>
#include <cassert>

// This test verifies that the number of devices detected inside and outside a target region matches.

int main(void) {
  int numDevives = 0;
  bool isHost = true;

  #pragma omp target map(from : isHost, numDevives)
  {
    numDevives = omp_get_num_devices();
    isHost = omp_is_initial_device();
  }

  assert(!isHost && "This test is supposed to execute with a device enabled.");

  // CHECK: omp_get_num_devices() returns the same on host and device? True.
  std::cout << "omp_get_num_devices() returns the same on host and device? ";
  std::cout << ((numDevives == omp_get_num_devices()) ? "True." : "False.");
  std::cout << std::endl;

  return isHost;
}
