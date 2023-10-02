//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKNEDS_COPY_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKNEDS_COPY_H

#include <__algorithm/copy.h>
#include <__algorithm/pstl_backends/cpu_backends/backend.h>
#include <__algorithm/pstl_backends/gpu_backends/backend.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__type_traits/is_execution_policy.h>
#include <__utility/terminate_on_exception.h>
#include <stdio.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ExecutionPolicy, class _ForwardIterator, class _ForwardOutIterator>
_LIBCPP_HIDE_FROM_ABI _ForwardOutIterator
__pstl_copy(__gpu_backend_tag, _ForwardIterator __first, _ForwardIterator __last, _ForwardOutIterator __result) {
  if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                __has_random_access_iterator_category_or_concept<_ForwardIterator>::value &&
                __has_random_access_iterator_category_or_concept<_ForwardOutIterator>::value
                //                __libcpp_is_contiguous_iterator<_ForwardIterator>::value &&
                //                __libcpp_is_contiguous_iterator<_ForwardOutIterator>::value
  ) {
    // XXX There's an even faster path that calls omp_target_memcpy, the slow path with __identity() will map your data
    // which is catastrophic for USM inputs in terms of performance
    std::__par_backend::__parallel_for_simd_2(__first, __last - __first, __result);
    return __result + (__last - __first);
  }
  std::abort();
  // If it is not safe to offload to the GPU, we rely on the CPU backend.
  return std::__pstl_transform<_ExecutionPolicy>(
      __cpu_backend_tag{},
      __first,
      __last,
      __result, //
      __identity());
}
_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKNEDS_COPY_H
