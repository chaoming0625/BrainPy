#ifndef _BRAINPY_MAT_MUL_MASK_KERNELS_H_
#define _BRAINPY_MAT_MUL_MASK_KERNELS_H_

#include "kernel_helpers_matmul.h"

namespace brainpy_lib {

    void mmm_8K_1x8x128x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void mmm_8K_1x16x64x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void mmm_8K_1x32x32x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void mmm_8K_1x64x16x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);

    void mmm_8K_4x8x128x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void mmm_8K_4x16x64x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void mmm_8K_4x32x32x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void mmm_8K_4x64x16x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);

    void mmm_4K_1x8x128x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void mmm_4K_1x16x64x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void mmm_4K_1x32x32x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void mmm_4K_1x64x16x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);

    void mmm_4K_4x8x128x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void mmm_4K_4x16x64x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void mmm_4K_4x32x32x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void mmm_4K_4x64x16x256(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);

}  // namespace brainpy_lib

#endif