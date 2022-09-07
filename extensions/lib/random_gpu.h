#ifndef _BRAINPY_RANDOM_TEST_KERNELS_H_
#define _BRAINPY_RANDOM_TEST_KERNELS_H_

#include <cstddef>
#include <cstdint>
#include "pybind11_kernel_helpers.h"
#include "kernel_helpers_gpu.h"

namespace brainpy_lib {
    struct RandomSampleDescriptor {
        std::uint32_t length;
        std::uint32_t seed;
    };

    pybind11::bytes build_rand_sample_descriptor(std::uint32_t length, std::uint32_t seed);

    void uniform(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void uniform2(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
}  // namespace brainpy_lib

#endif