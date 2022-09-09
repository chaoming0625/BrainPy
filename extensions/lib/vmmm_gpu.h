#ifndef _BRAINPY_VECTOR_MATMUL_MASKED_KERNELS_H_
#define _BRAINPY_VECTOR_MATMUL_MASKED_KERNELS_H_

#include <cstddef>
#include <cstdint>
#include "kernel_helpers_matmul.h"

namespace brainpy_lib {

    void vector_matmul_mask(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);

}  // namespace brainpy_lib

#endif