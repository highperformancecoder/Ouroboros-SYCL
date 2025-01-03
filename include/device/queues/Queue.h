#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#pragma once

#include "Definitions.h"

namespace Ouro
{
  class IndexQueue
  {
  public:
    template <class Desc>
    __dpct_inline__ void init(const Desc&);

    __dpct_inline__ bool enqueue(index_t i);

    template <int CHUNK_SIZE>
    __dpct_inline__ bool enqueueClean(index_t i, index_t *chunk_data_ptr);

    __dpct_inline__ int dequeue(index_t &element);

    void resetQueue();

    index_t* queue_;
    int count_{ 0 };
    unsigned int front_{ 0 };
    unsigned int back_{ 0 };
    int size_{ 0 };
  };
}
