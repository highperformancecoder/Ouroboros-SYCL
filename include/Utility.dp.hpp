#pragma once
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "Utility.h"
#include <time.h>

namespace Ouro
{
  template <class T> using Atomic=sycl::atomic_ref<T,sycl::memory_order::relaxed,sycl::memory_scope::device>;

  struct DummyStream
  {
    template <class T>
    const DummyStream& operator<<(T) const {return *this;}
  };
  
  template<int Rank=1,class S=DummyStream>
  struct SyclDesc
  {
    sycl::nd_item<Rank> item;
    S out;
    SyclDesc(const sycl::nd_item<Rank>& item, const S& out={}): item(item), out(out) {}
  };

  template <class M>
  class ThreadAllocator
  {
    SyclDesc<1,sycl::stream> m_desc;
    M& m;
  public:
    ThreadAllocator(const sycl::nd_item<1>& item, const sycl::stream& out, M& m):
      m_desc(item,out), m(m) {}
    const SyclDesc<1,sycl::stream>& desc() const {return m_desc;}
    void* malloc(size_t sz) {return m.malloc(m_desc,sz);}
    void free(void* p) {m.free(m_desc,p);}
  };
  
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

  //#if defined(DPCT_COMPATIBILITY_TEMP) && (DPCT_COMPATIBILITY_TEMP >= 700)
#if 0
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
    return Atomic<unsigned>(*ptr).fetch_add(1);
  }
#endif



  template <class T, class U> T atomicAdd(T* x, U v)
  {return Atomic<T>(*x).fetch_add(v);}

  template <class T, class U> T atomicSub(T* x, U v)
  {return Atomic<T>(*x).fetch_sub(v);}

  template <class T, class U> T atomicAnd(T* x, U v)
  {return Atomic<T>(*x).fetch_and(v);}

  template <class T, class U> T atomicOr(T* x, U v)
  {return Atomic<T>(*x).fetch_or(v);}

  template <class T, class U> T atomicMax(T* x, U v)
  {return Atomic<T>(*x).fetch_max(v);}

  template <class T, class U> T atomicExch(T* x, U v)
  {return Atomic<T>(*x).exchange(v);}

  template <class T> T atomicCAS(T* x, T expected, T desired)
  {
    Atomic<T>(*x).compare_exchange_strong(expected,desired);
    return expected; // value updated to previous value of x by above
  }

}
