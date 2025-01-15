#pragma once
#include <sycl/sycl.hpp>
#include "Queue.h"
#include "Utility.dp.hpp"

namespace Ouro
{
  inline void IndexQueue::resetQueue()
  {
    count_ = 0;
    front_ = 0;
    back_ = 0;
  }

  template <class Desc>
  inline void IndexQueue::init(const Desc& d)
  {
    for (int i = d.item.get_global_linear_id();
         i < size_;
         i += d.item.get_global_range().size())
      {
        queue_[i] = DeletionMarker<index_t>::val;
      }
  }

  inline bool IndexQueue::enqueue(index_t i)
  {
    int fill = atomicAdd(&count_, 1);
    if (fill < static_cast<int>(size_))
      {
        //we have to wait in case there is still something in the spot
        // note: as the filllevel could be increased by this thread, we are certain that the spot will become available
        unsigned int pos = atomicAdd(&back_, 1) % size_;
        while (atomicCAS(queue_ + pos, DeletionMarker<index_t>::val, i) != DeletionMarker<index_t>::val)
          Ouro::sleep();
			
        return true;
      }
    else
      {
        //__trap(); //no space to enqueue -> fail
        return false;
      }
  }

  template <int CHUNK_SIZE>
  inline bool IndexQueue::enqueueClean(index_t i,
                                                index_t *chunk_data_ptr)
  {
    for(auto i = 0U; i < (CHUNK_SIZE / (sizeof(index_t))); ++i)
      {
        //atomicExch(&chunk_data_ptr[i], DeletionMarker<index_t>::val);
        Ouro::Atomic<unsigned>{chunk_data_ptr[i]}=DeletionMarker<index_t>::val;
      }

    /*
      DPCT1078:0: Consider replacing memory_order::acq_rel with
      memory_order::seq_cst for correctness if strong memory order
      restrictions are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel,
                       sycl::memory_scope::work_group);

    // Enqueue now
    return enqueue(i);
  }

  inline int IndexQueue::dequeue(index_t &element)
  {
    int readable = atomicSub(&count_, 1);
    if (readable <= 0)
      {
        //dequeue not working is a common case
        atomicAdd(&count_, 1);
        return FALSE;
      }
    unsigned int pos = atomicAdd(&front_, 1) % size_;
    while ((element = atomicExch(queue_ + pos, DeletionMarker<index_t>::val)) == DeletionMarker<index_t>::val)
      Ouro::sleep();
    return TRUE;
  }
}
