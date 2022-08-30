#ifndef _BRAINPY_MAT_MUL_KERNELS_H_
#define _BRAINPY_MAT_MUL_KERNELS_H_

#include <cstddef>
#include <cstdint>
#include "pybind11_kernel_helpers.h"
#include "kernel_helpers_gpu.h"

namespace brainpy_lib {

    struct MatMulDescriptor {
        std::uint32_t m;
        std::uint32_t k;
        std::uint32_t n;
        std::uint32_t seed;
        float p;
    };

    pybind11::bytes build_matmul_descriptor(std::uint32_t m,
                                            std::uint32_t k,
                                            std::uint32_t n,
                                            std::uint32_t seed,
                                            float p);

    void vector_matmul_mask(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);

}  // namespace brainpy_lib

#endif