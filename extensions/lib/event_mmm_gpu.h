#ifndef _BRAINPY_EVENT_MAT_MUL_MASK_KERNELS_H_
#define _BRAINPY_EVENT_MAT_MUL_MASK_KERNELS_H_

#include "kernel_helpers_matmul.h"
#include<cmath>

namespace brainpy_lib {

    void event_mmm_fp_v1_4x64x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void event_mmm_fp_v1_8x32x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void event_mmm_fp_v1_16x16x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void event_mmm_fp_v1_32x8x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);


    void event_mmm_fp_v2_4x64x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void event_mmm_fp_v2_8x32x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void event_mmm_fp_v2_16x16x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void event_mmm_fp_v2_32x8x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);


}  // namespace brainpy_lib

#endif