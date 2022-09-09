#ifndef _BRAINPY_EVENT_MAT_MUL_MASK_KERNELS_H_
#define _BRAINPY_EVENT_MAT_MUL_MASK_KERNELS_H_

#include "kernel_helpers_matmul.h"
#include<cmath>

namespace brainpy_lib {

    void event_mmm_8K_1x8x128x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void event_mmm_8K_4x8x128x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);


}  // namespace brainpy_lib

#endif