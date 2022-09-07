#ifndef _BRAINPY_MAT_MUL_MASK_KERNELS_H_
#define _BRAINPY_MAT_MUL_MASK_KERNELS_H_

#include "kenerl_helpers_matmul.h"

namespace brainpy_lib {

    void mmm_8K_8x256x512(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void mmm_8K_8x128x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void mmm_8K_16x128x512(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void mmm_8K_16x64x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void mmm_8K_32x64x512(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void mmm_8K_32x32x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void event_mmm_8K_16x64x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);

}  // namespace brainpy_lib

#endif