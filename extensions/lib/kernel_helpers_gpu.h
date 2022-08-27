// This header is not specific to our application and you'll probably want something like this
// for any extension you're building. This includes the infrastructure needed to serialize
// descriptors that are used with the "opaque" parameter of the GPU custom call. In our example
// we'll use this parameter to pass the size of our problem.

#ifndef _BRAINPYLIB_KERNEL_HELPERS_CUDA_H_
#define _BRAINPYLIB_KERNEL_HELPERS_CUDA_H_


#include <cstdint>
#include <cuda_runtime_api.h>

namespace brainpy_lib {
    // error handling //
    static void ThrowIfError(cudaError_t error) {
        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }
    }


    __device__ __forceinline__

    uint32_t smem_u32addr(const void *smem_ptr) {
        uint32_t addr;
        asm ("{.reg .u64 u64addr;\n"
             " cvta.to.shared.u64 u64addr, %1;\n"
             " cvt.u32.u64 %0, u64addr;}\n"
                : "=r"(addr)
                : "l"(smem_ptr)
                );

        return addr;
    }

    __device__ __forceinline__

    void ldg32_nc(float &reg, const void *ptr, bool guard) {
        asm volatile (
                "{.reg .pred p;\n"
                " setp.ne.b32 p, %2, 0;\n"
                #if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
                " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
                #else
                " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
                : "=f"(reg)
                : "l"(ptr), "r"((int) guard)
                );
    }

    __device__ __forceinline__

    void ldg32_nc_0(float &reg, const void *ptr, bool guard) {
        asm volatile (
                "{.reg .pred p;\n"
                " setp.ne.b32 p, %2, 0;\n"
                " @!p mov.b32 %0, 0;\n"
                #if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && __CUDA_ARCH__ >= 750
                " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
                #else
                " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
                : "=f"(reg)
                : "l"(ptr), "r"((int) guard)
                );
    }

    // load float32
    __device__ __forceinline__

    void stg32(const float &reg, void *ptr, bool guard) {
        asm volatile (
                "{.reg .pred p;\n"
                " setp.ne.b32 p, %2, 0;\n"
                " @p st.global.f32 [%0], %1;}\n"
                : : "l"(ptr), "f"(reg), "r"((int) guard)
                );
    }

    // load float32 * 4
    __device__ __forceinline__
    void lds128(float &reg0, float &reg1,
                float &reg2, float &reg3,
                const uint32_t &addr) {
        asm volatile (
                "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
                : "r"(addr)
                );
    }

    // store float32
    __device__ __forceinline__

    void sts32(const float &reg, const uint32_t &addr) {
        asm volatile (
                "st.shared.f32 [%0], %1;\n"
                : : "r"(addr), "f"(reg)
                );
    }

    // store float32 * 4
    __device__ __forceinline__

    void sts128(const float &reg0, const float &reg1,
                const float &reg2, const float &reg3,
                const uint32_t &addr) {
        asm volatile (
                "st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
                : : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3)
                );
    }


}  // namespace brainpy_lib

#endif
