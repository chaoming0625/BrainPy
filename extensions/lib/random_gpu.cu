// This file contains the GPU implementation of our op. It's a pretty typical CUDA kernel
// and I make no promises about the quality of the code or the choices made therein, but
// it should get the point across.

#include "random_gpu.h"
#include "event_sum_gpu.h"
#include "curand_kernel.h"

namespace brainpy_lib {

    __global__ void uniform_kernel(float *O, std::uint32_t size, int seed) {
        std::uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;

        if (x >= size) {
            return;
        }
        curandState state;
        curand_init(seed + blockIdx.x, 0, 0, &state);
        O[x] = curand_uniform(&state);
    }

    __global__ void uniform_kernel2(float *O, std::uint32_t size, curandState *rngs) {
        std::uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;

        if (x >= size) {
            return;
        }
        curandState state = rngs[blockIdx.x];
        O[x] = curand_uniform(&state);
        O[x + size] = curand_uniform(&state);
        O[x + size * 2] = curand_uniform(&state);
//        rngs[blockIdx.x] = state;
    }


    void uniform(cudaStream_t stream,
                 void **buffers,
                 const char *opaque,
                 std::size_t opaque_len) {

        // size
        const RandomSampleDescriptor &d = *UnpackDescriptor<RandomSampleDescriptor>(opaque, opaque_len);
        const std::uint32_t len = d.length;
        const std::uint32_t seed = d.seed;

        // input and output data
        float *O = reinterpret_cast<float *>(buffers[0]);

        // call kernel
        cudaMemset(O, 0, sizeof(float) * len);
        uniform_kernel<<<(len + 31) / 32, 32, 0, stream>>>(O, len, seed);
        ThrowIfError(cudaGetLastError());
    }


    void uniform2(cudaStream_t stream,
                 void **buffers,
                 const char *opaque,
                 std::size_t opaque_len) {

        // size
        const RandomSampleDescriptor &d = *UnpackDescriptor<RandomSampleDescriptor>(opaque, opaque_len);
        const std::uint32_t len = d.length;

        // input and output data
        curandState *rngs = reinterpret_cast<curandState *>(buffers[0]);
        float *O = reinterpret_cast<float *>(buffers[1]);

        // call kernel
        cudaMemset(O, 0, sizeof(float) * len);
        uniform_kernel2<<<(len + 31) / 32, 32, 0, stream>>>(O, len, rngs);
        ThrowIfError(cudaGetLastError());
    }

    // Descriptors
    pybind11::bytes build_rand_sample_descriptor(
            std::uint32_t length,
            std::uint32_t seed
    ) {
        return PackDescriptor(RandomSampleDescriptor{length, seed});
    }


}  // namespace brainpylib
