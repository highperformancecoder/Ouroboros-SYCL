#include <sycl/sycl.hpp>
//#include <dpct/dpct.hpp>
#pragma once

#include "Definitions.h"

namespace Ouro
{
  class IndexQueue
  {
  public:
    template <class Desc>
    inline void init(const Desc&);

    inline bool enqueue(index_t i);

    template <int CHUNK_SIZE>
    inline bool enqueueClean(index_t i, index_t *chunk_data_ptr);

    inline int dequeue(index_t &element);

    void resetQueue();

    index_t* queue_;
    int count_{ 0 };
    unsigned int front_{ 0 };
    unsigned int back_{ 0 };
    int size_{ 0 };
  };
}
