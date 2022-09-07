#ifndef _BRAINPY_MASKED_MATMUL_KERNELS_H_
#define _BRAINPY_MASKED_MATMUL_KERNELS_H_

#include "kenerl_helpers_matmul.h"

namespace brainpy_lib {

    void masked_matmul(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);

}  // namespace brainpy_lib

#endif