//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_OMP_OFFLOAD_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_OMP_OFFLOAD_H

#include <__assert>
#include <__config>
#include <__iterator/iterator_traits.h>
#include <stdio.h>
//#include <omp.h>
#include <cassert>

extern "C" int omp_is_initial_device(void);

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __par_backend {
inline namespace __omp_backend {

//===----------------------------------------------------------------------===//
// The following four functions can be used to map contiguous array sections to
// and from the device. For now, they are simple overlays of the OpenMP pragmas,
// but they should be updated when adding support for other iterator types.
//===----------------------------------------------------------------------===//

template <class _Iterator, class _DifferenceType>
_LIBCPP_HIDE_FROM_ABI void
__omp_map_to([[maybe_unused]] const _Iterator __p, [[maybe_unused]] const _DifferenceType __len) noexcept {
  static_assert(__libcpp_is_contiguous_iterator<_Iterator>::value);
#  pragma omp target enter data map(to : __p[0 : __len])
}

template <class _Iterator, class _DifferenceType>
_LIBCPP_HIDE_FROM_ABI void
__omp_map_from([[maybe_unused]] const _Iterator __p, [[maybe_unused]] const _DifferenceType __len) noexcept {
  static_assert(__libcpp_is_contiguous_iterator<_Iterator>::value);
#  pragma omp target exit data map(from : __p[0 : __len])
}

template <class _Iterator, class _DifferenceType>
_LIBCPP_HIDE_FROM_ABI void
__omp_map_alloc([[maybe_unused]] const _Iterator __p, [[maybe_unused]] const _DifferenceType __len) noexcept {
  static_assert(__libcpp_is_contiguous_iterator<_Iterator>::value);
#  pragma omp target enter data map(alloc : __p[0 : __len])
}

template <class _Iterator, class _DifferenceType>
_LIBCPP_HIDE_FROM_ABI void
__omp_map_release([[maybe_unused]] const _Iterator __p, [[maybe_unused]] const _DifferenceType __len) noexcept {
  static_assert(__libcpp_is_contiguous_iterator<_Iterator>::value);
#  pragma omp target exit data map(release : __p[0 : __len])
}

//===----------------------------------------------------------------------===//
// Getting the value of a function pointer from the target region
//===----------------------------------------------------------------------===//

// template <class _Function,class = void>
// _LIBCPP_HIDE_FROM_ABI _Function
// __omp_get_device_ptr(const _Function __f) noexcept {
//   printf("This is not a function pointer\n");
//   printf("This is not a function pointer\n");
//   printf("This is not a function pointer\n");
//   return __f;
// }
// // Specialization for function pointers
// template <class _Function,__enable_if_t<std::is_function<_Function>::value,int> = 0 >
// _LIBCPP_HIDE_FROM_ABI _Function *
// __omp_get_device_ptr(const _Function * __f) noexcept {
//   printf("This is a function pointer\n");
//   _Function * __d;
// #  pragma omp target map(from:__d)
//   __d = __f;
//   return __d;
// }

template <class _Function,__enable_if_t<!(std::is_pointer<_Function>::value && std::is_function<typename std::remove_pointer<_Function>::type>::value ),int> = 0 >
_LIBCPP_HIDE_FROM_ABI _Function
__omp_get_device_ptr(_Function __f) noexcept {
  printf("This is not a function pointer\n");
  return __f;
}

// Specialization for function pointers
template <class _Function,__enable_if_t<std::is_pointer<_Function>::value && std::is_function<typename std::remove_pointer<_Function>::type>::value ,int> = 0 >
_LIBCPP_HIDE_FROM_ABI _Function
__omp_get_device_ptr(_Function __f) noexcept {
  printf("This is a function pointer\n");
  _Function __d;
  [[maybe_unused]] bool __on_device = false;
  printf("host: %p\n",(uintptr_t)__f);
#pragma omp target map(always,from : __d,__on_device)
  {
    __d = __f;
    __on_device = !omp_is_initial_device();
  }
  printf("device: %p\n",(uintptr_t)__d);
  assert(__on_device && "OpenMP target region not executed on device.");
  assert((__d != __f) && "OpenMP could not get device address of host pointer.");
  return __d;
}

} // namespace __omp_backend
} // namespace __par_backend

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_OMP_OFFLOAD_H
