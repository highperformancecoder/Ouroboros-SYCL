#pragma once
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "Utility.h"
#include <time.h>

namespace Ouro
{
  template <class T> using Atomic=sycl::atomic_ref<T,sycl::memory_order::relaxed,sycl::memory_scope::device>;

  // replacement for CUDA atomicCAS
  template <class T> inline T atomicCAS(T& address, T compare, T val)
  {
    Atomic<T>(address).compare_exchange_strong(compare,val,sycl::memory_order::relaxed,sycl::memory_scope::device);
    return compare;
  }
  
  __dpct_inline__ unsigned int ldg_cg(const unsigned int *src)
  {
    unsigned int dest{ 0 };
#ifdef DPCT_COMPATIBILITY_TEMP
    /*
      DPCT1053:0: Migration of device assembly code is not supported.
    */
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(dest) : "l"(src));
#endif
    return dest;
  }

  __dpct_inline__ int ldg_cg(const int *src)
  {
    int dest{ 0 };
#ifdef DPCT_COMPATIBILITY_TEMP
    /*
      DPCT1053:1: Migration of device assembly code is not
      supported.
    */
    asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(dest) : "l"(src));
#endif
    return dest;
  }

  __dpct_inline__ unsigned long long ldg_cg(const unsigned long long *src)
  {
    unsigned long long dest{0};
#ifdef DPCT_COMPATIBILITY_TEMP
    /*
      DPCT1053:2: Migration of device assembly code is not supported.
    */
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(dest) : "l"(src));
#endif
    return dest;
  }

  __dpct_inline__ const unsigned int &stg_cg(unsigned int *dest,
                                             const unsigned int &src)
  {
#ifdef DPCT_COMPATIBILITY_TEMP
    /*
      DPCT1053:3: Migration of device assembly code is not supported.
    */
    asm volatile("st.global.cg.u32 [%0], %1;" : : "l"(dest), "r"(src));
#endif
    return src;
  }

  /*
    DPCT1052:8: SYCL does not support the member access for a volatile qualified
    vector type. The volatile qualifier was removed. You may need to rewrite the
    code.
  */
  __dpct_inline__ void store(sycl::uint4 *dest, const sycl::uint4 &src)
  {
#ifdef DPCT_COMPATIBILITY_TEMP
    /*
      DPCT1053:4: Migration of device assembly code is not supported.
    */
    asm("st.volatile.v4.u32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(dest), "r"(src.x()), "r"(src.y()), "r"(src.z()),
          "r"(src.w()));
#endif
  }

  /*
    DPCT1052:9: SYCL does not support the member access for a volatile qualified
    vector type. The volatile qualifier was removed. You may need to rewrite the
    code.
  */
  __dpct_inline__ void store(sycl::uint2 *dest, const sycl::uint2 &src)
  {
#ifdef DPCT_COMPATIBILITY_TEMP
    /*
      DPCT1053:5: Migration of device assembly code is not supported.
    */
    asm("st.volatile.v2.u32 [%0], {%1, %2};"
        :
        : "l"(dest), "r"(src.x()), "r"(src.y()));
#endif
  }

  static __dpct_inline__ int lane_id(const sycl::nd_item<3> &item_ct1)
  {
    return item_ct1.get_local_id(2) & (WARP_SIZE - 1);
  }

  __dpct_inline__ void sleep(unsigned int factor = 1)
  {
#ifdef DPCT_COMPATIBILITY_TEMP
    //#if (DPCT_COMPATIBILITY_TEMP >= 700)
    //                //__nanosleep(SLEEP_TIME);
    //                /*
    //                DPCT1008:10: __nanosleep function is not defined in SYCL. This
    //                is a hardware-specific feature. Consult with your hardware
    //                vendor to find a replacement.
    //                */
    //                __nanosleep(SLEEP_TIME * factor);
    //#else
    //__threadfence();
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);
    //	#endif
#endif
  }

#ifdef DPCT_COMPATIBILITY_TEMP
#if (DPCT_COMPATIBILITY_TEMP >= 700)
  __dpct_inline__ int atomicAggInc(unsigned int *ptr,
                                   const sycl::nd_item<3> &item_ct1)
  {
    /*
      DPCT1086:6: __activemask() is migrated to 0xffffffff. You may
      need to adjust the code.
    */
    int mask = dpct::match_any_over_sub_group(
                                              item_ct1.get_sub_group(), 0xffffffff,
                                              reinterpret_cast<unsigned long long>(ptr));
    int leader = dpct::ffs<int>(mask) - 1;
    int res = 0;
    if (lane_id(item_ct1) == leader)
      res = dpct::atomic_fetch_add<
        sycl::access::address_space::generic_space>(
                                                    ptr, sycl::popcount(mask));
    /*
      DPCT1023:7: The SYCL sub-group does not support mask options for
      dpct::select_from_sub_group. You can specify
      "--use-experimental-features=masked-sub-group-operation" to use
      the experimental helper function to migrate __shfl_sync.
    */
    res = dpct::select_from_sub_group(item_ct1.get_sub_group(), res, leader);
    return res + sycl::popcount(mask & ((1 << lane_id(item_ct1)) - 1));
  }
#else
  __dpct_inline__ int atomicAggInc(unsigned int *ptr)
  {
    return ++Ouro::Atomic<unsigned>(*ptr);
  }
#endif
#else
  __dpct_inline__ int atomicAggInc(unsigned int *ptr)
  {
    auto val = *ptr;
    *ptr += 1;
    return val;
  }
#endif

}

