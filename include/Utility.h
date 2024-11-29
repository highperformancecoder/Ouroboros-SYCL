#pragma once

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <string>

#include "Definitions.h"
#include "Parameters.h"

namespace Ouro
{
  // ##############################################################################################################################################
  //
  static inline void HandleError(dpct::err0 err, const char *string,
                                 const char *file, int line) {
  }

  // ##############################################################################################################################################
  //
  static inline void HandleError(const char *file,
                                 int line) {
    /*
      DPCT1010:13: SYCL uses exceptions to report errors and does not use the
      error codes. The call was replaced with 0. You need to rewrite this
      code.
    */
    auto err = 0;
  }

#define HANDLE_ERROR( err ) (Ouro::HandleError( err, "", __FILE__, __LINE__ ))
#define HANDLE_ERROR_S( err , string) (Ouro::HandleError( err, string, __FILE__, __LINE__ ))

  // ##############################################################################################################################################
  //
  static inline void DEBUG_checkKernelError(const char* message = nullptr)
  {
    if (debug_enabled)
      {
        /*
          DPCT1010:15: SYCL uses exceptions to report errors and does not
          use the error codes. The call was replaced with 0. You need to
          rewrite this code.
        */
        HANDLE_ERROR(0);
        HANDLE_ERROR(DPCT_CHECK_ERROR(
                                      dpct::get_current_device().queues_wait_and_throw()));
        if (printDebug && message)
          printf("%s\n", message);
      }
  }

  void queryAndPrintDeviceProperties();


  // ##############################################################################################################################################
  //
  void inline start_clock(dpct::event_ptr &start, dpct::event_ptr &end)
  {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    HANDLE_ERROR(DPCT_CHECK_ERROR(start = new sycl::event()));
    HANDLE_ERROR(DPCT_CHECK_ERROR(end = new sycl::event()));
    /*
      DPCT1024:16: The original code returned the error code that was further
      consumed by the program logic. This original code was replaced with 0.
      You may need to rewrite the program logic consuming the error code.
    */
    HANDLE_ERROR(DPCT_CHECK_ERROR(dpct::sync_barrier(start, &q_ct1)));
  }

  // ##############################################################################################################################################
  //
  float inline end_clock(dpct::event_ptr &start, dpct::event_ptr &end)
  {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    float time;
    /*
      DPCT1024:17: The original code returned the error code that was further
      consumed by the program logic. This original code was replaced with 0.
      You may need to rewrite the program logic consuming the error code.
    */
    HANDLE_ERROR(DPCT_CHECK_ERROR(dpct::sync_barrier(end, &q_ct1)));
    HANDLE_ERROR(DPCT_CHECK_ERROR(end->wait_and_throw()));
    HANDLE_ERROR(DPCT_CHECK_ERROR(
                                  time = (end->get_profiling_info<
                                          sycl::info::event_profiling::command_end>() -
                                          start->get_profiling_info<
                                          sycl::info::event_profiling::command_start>()) /
                                  1000000.0f));
    HANDLE_ERROR(DPCT_CHECK_ERROR(dpct::destroy_event(start)));
    HANDLE_ERROR(DPCT_CHECK_ERROR(dpct::destroy_event(end)));

    // Returns ms
    return time;
  }

  // ##############################################################################################################################################
  //
  static constexpr __dpct_inline__ unsigned long long
  create2Complement(unsigned long long value)
  {
    return ~(value) + 1ULL;
  }

  // ##############################################################################################################################################
  //
  template <typename T>
  static constexpr bool isPowerOfTwo(T n) 
  {
    return (n & (n - 1)) == 0;
  }

  // ##############################################################################################################################################
  //
  template <typename T> __dpct_inline__ T divup(T a, T b)
  {
    return (a + b - 1) / b;
  }

  // ##############################################################################################################################################
  //
  template <typename T, typename O> constexpr __dpct_inline__ T divup(T a, O b)
  {
    return (a + b - 1) / b;
  }

  // ##############################################################################################################################################
  //
  template <typename T>
  constexpr __dpct_inline__ T alignment(const T size,
                                        size_t alignment = CACHELINE_SIZE)
  {
    return divup<T, size_t>(size, alignment) * alignment;
  }

  // ##############################################################################################################################################
  //
  template <typename T> constexpr __dpct_inline__ size_t sizeofInBits()
  {
    return sizeof(T) * BYTE_SIZE;
  }

  // ##############################################################################################################################################
  //
  template<unsigned int X, int Completed = 0>
  struct static_clz
  {
    static const int value = (X & 0x80000000) ? Completed : static_clz< (X << 1), Completed + 1 >::value;
  };

  // ##############################################################################################################################################
  //
  template<unsigned int X>
  struct static_clz<X, 32>
  {
    static const int value = 32;
  };

  // ##############################################################################################################################################
  //
  static constexpr int cntlz(unsigned int x)
  {
    if (x == 0) return 32;
    int n = 0;
    if (x <= 0x0000FFFF) { n = n + 16; x = x << 16; }
    if (x <= 0x00FFFFFF) { n = n + 8; x = x << 8; }
    if (x <= 0x0FFFFFFF) { n = n + 4; x = x << 4; }
    if (x <= 0x3FFFFFFF) { n = n + 2; x = x << 2; }
    if (x <= 0x7FFFFFFF) { n = n + 1; x = x << 1; }
    return n;
  }

  // ##############################################################################################################################################
  //
  static inline void printTestcaseSeparator(const std::string& header)
  {
    printf("%s", break_line_purple_s);
    printf("#%105s\n", "#");
    printf("###%103s\n", "###");
    printf("#####%101s\n", "#####");
    printf("#######%99s\n", "#######");
    printf("#########%55s%42s\n", header.c_str(), "#########");
    printf("#######%99s\n", "#######");
    printf("#####%101s\n", "#####");
    printf("###%103s\n", "###");
    printf("#%105s\n", "#");
    printf("%s", break_line_purple_e);
  }

  static constexpr char PBSTR[] = "##############################################################################################################";
  static constexpr int PBWIDTH = 99;

  // ##############################################################################################################################################
  //
  static inline void printProgressBar(const double percentage)
  {
    auto val = static_cast<int>(percentage * 100);
    auto lpad = static_cast<int>(percentage * PBWIDTH);
    auto rpad = PBWIDTH - lpad;
#ifdef WIN32
    printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
#else
    printf("\r\033[0;35m%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
#endif
    fflush(stdout);
  }

  // ##############################################################################################################################################
  //
  static inline void printProgressBarEnd()
  {
#ifdef WIN32
    printf("\n");
#else
    printf("\033[0m\n");
#endif
    fflush(stdout);
  }

  // ##############################################################################################################################################
  //
  template <typename Data>
  void updateDataHost(Data& data)
  {
    HANDLE_ERROR(DPCT_CHECK_ERROR(
                                  dpct::get_in_order_queue()
                                  .memcpy(&data, data.d_memory, sizeof(Data))
                                  .wait()));
  }

  // ##############################################################################################################################################
  //
  template <typename Data>
  void updateDataDevice(Data& data)
  {
    HANDLE_ERROR(DPCT_CHECK_ERROR(
                                  dpct::get_in_order_queue()
                                  .memcpy(data.d_memory, &data, sizeof(Data))
                                  .wait()));
  }

  // ##############################################################################################################################################
  //
  template <typename T, typename SizeType>
  static constexpr __dpct_inline__ T modPower2(T value, SizeType size)
  {
    return value & (size - 1);
  }

  // ##############################################################################################################################################
  //
  template <unsigned int size>
  static constexpr __dpct_inline__ unsigned int
  modPower2(const unsigned int value)
  {
    static_assert(isPowerOfTwo(size), "ModPower2 used with non-power of 2");
    return value & (size - 1);
  }

  // ##############################################################################################################################################
  // Error Codes
  using ErrorType = unsigned int;
  enum class ErrorCodes
    {
      NO_ERROR,
      OUT_OF_CUDA_MEMORY,
      OUT_OF_CHUNK_MEMORY,
      CHUNK_ENQUEUE_ERROR
    };

  // ##############################################################################################################################################
  //
  template <typename DataType, ErrorCodes Error>
  struct ErrorVal;

  // ##############################################################################################################################################
  //
  template <typename DataType>
  struct ErrorVal<DataType, ErrorCodes::NO_ERROR>
  {
    static constexpr DataType value{ 0 };

    static constexpr __dpct_inline__ void setError(ErrorType &error)
    {
#ifdef DPCT_COMPATIBILITY_TEMP
      atomicOr(&error, value);
#else
      error |= value;
#endif
    }

    static constexpr __dpct_inline__ bool checkError(ErrorType &error)
    {
      return error != 0;
    }

    static constexpr __dpct_inline__ void print()
    {
      /*
        DPCT1040:0: Use sycl::stream instead of printf if your
        code is used on the device.
      */
      printf("No Error\n");
    }
  };

  // ##############################################################################################################################################
  //
  template <typename DataType>
  struct ErrorVal<DataType, ErrorCodes::OUT_OF_CUDA_MEMORY>
  {
    static constexpr DataType value{ 1 << 0 };

    static constexpr __dpct_inline__ void setError(ErrorType &error)
    {
#ifdef DPCT_COMPATIBILITY_TEMP
      atomicOr(&error, value);
#else
      error |= value;
#endif
    }

    static constexpr __dpct_inline__ bool checkError(ErrorType &error)
    {
      return error & value;
    }

    static constexpr __dpct_inline__ void print()
    {
      /*
        DPCT1040:1: Use sycl::stream instead of printf if your
        code is used on the device.
      */
      printf("Out of CUDA Memory Error\n");
    }
  };

  // ##############################################################################################################################################
  //
  template <typename DataType>
  struct ErrorVal<DataType, ErrorCodes::OUT_OF_CHUNK_MEMORY>
  {
    static constexpr DataType value{ 1 << 1 };

    static constexpr __dpct_inline__ void setError(ErrorType &error)
    {
#ifdef DPCT_COMPATIBILITY_TEMP
      atomicOr(&error, value);
#else
      error |= value;
#endif
    }

    static constexpr __dpct_inline__ bool checkError(ErrorType &error)
    {
      return error & value;
    }

    static constexpr __dpct_inline__ void print()
    {
      /*
        DPCT1040:2: Use sycl::stream instead of printf if your
        code is used on the device.
      */
      printf("Out of Chunk Memory Error\n");
    }
  };

  // ##############################################################################################################################################
  //
  template <typename DataType>
  struct ErrorVal<DataType, ErrorCodes::CHUNK_ENQUEUE_ERROR>
  {
    static constexpr DataType value{ 1 << 2 };

    static constexpr __dpct_inline__ void setError(ErrorType &error)
    {
#ifdef DPCT_COMPATIBILITY_TEMP
      atomicOr(&error, value);
#else
      error |= value;
#endif
    }

    static constexpr __dpct_inline__ bool checkError(ErrorType &error)
    {
      return error & value;
    }

    static constexpr __dpct_inline__ void print()
    {
      /*
        DPCT1040:3: Use sycl::stream instead of printf if your
        code is used on the device.
      */
      printf("Chunk Enqueue Error\n");
    }
  };
}
