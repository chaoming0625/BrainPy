// This file defines the Python interface to the XLA custom call implemented on the GPU.
// Like in cpu_ops.cc, we export a separate capsule for each supported dtype, but we also
// include one extra method "build_kepler_descriptor" to generate an opaque representation
// of the problem size that will be passed to the op. The actual implementation of the
// custom call can be found in kernels.cc.cu.

#include <cstddef>
#include "pybind11_kernel_helpers.h"
#include "event_sum_gpu.h"
#include "atomic_sum_gpu.h"
#include "atomic_prod_gpu.h"
#include "vmmm_gpu.h"
#include "mmm_gpu.h"
#include "mat_mtp_mask_gpu.h"
#include "random_gpu.h"
#include "kenerl_helpers_matmul.h"

using namespace brainpy_lib;

namespace {
    pybind11::dict Registrations() {
        pybind11::dict dict;

        // homogeneous event_sum
        dict["gpu_event_sum_homo_f32_i32"] = EncapsulateFunction(gpu_event_sum_homo_f32_i32);
        dict["gpu_event_sum_homo_f32_i64"] = EncapsulateFunction(gpu_event_sum_homo_f32_i64);
//        dict["gpu_event_sum_homo_f64_i32"] = EncapsulateFunction(gpu_event_sum_homo_f64_i32);
//        dict["gpu_event_sum_homo_f64_i64"] = EncapsulateFunction(gpu_event_sum_homo_f64_i64);
        // heterogeneous event_sum
        dict["gpu_event_sum_heter_f32_i32"] = EncapsulateFunction(gpu_event_sum_heter_f32_i32);
        dict["gpu_event_sum_heter_f32_i64"] = EncapsulateFunction(gpu_event_sum_heter_f32_i64);
//        dict["gpu_event_sum_heter_f64_i32"] = EncapsulateFunction(gpu_event_sum_heter_f64_i32);
//        dict["gpu_event_sum_heter_f64_i64"] = EncapsulateFunction(gpu_event_sum_heter_f64_i64);

        // homogeneous event_sum2
        dict["gpu_event_sum2_homo_f32_i32"] = EncapsulateFunction(gpu_event_sum2_homo_f32_i32);
        dict["gpu_event_sum2_homo_f32_i64"] = EncapsulateFunction(gpu_event_sum2_homo_f32_i64);
//        dict["gpu_event_sum2_homo_f64_i32"] = EncapsulateFunction(gpu_event_sum2_homo_f64_i32);
//        dict["gpu_event_sum2_homo_f64_i64"] = EncapsulateFunction(gpu_event_sum2_homo_f64_i64);
        // heterogeneous event_sum2
        dict["gpu_event_sum2_heter_f32_i32"] = EncapsulateFunction(gpu_event_sum2_heter_f32_i32);
        dict["gpu_event_sum2_heter_f32_i64"] = EncapsulateFunction(gpu_event_sum2_heter_f32_i64);
//        dict["gpu_event_sum2_heter_f64_i32"] = EncapsulateFunction(gpu_event_sum2_heter_f64_i32);
//        dict["gpu_event_sum2_heter_f64_i64"] = EncapsulateFunction(gpu_event_sum2_heter_f64_i64);

        // homogeneous atomic_sum
        dict["gpu_atomic_sum_homo_f32_i32"] = EncapsulateFunction(gpu_atomic_sum_homo_f32_i32);
        dict["gpu_atomic_sum_homo_f32_i64"] = EncapsulateFunction(gpu_atomic_sum_homo_f32_i64);
//        dict["gpu_atomic_sum_homo_f64_i32"] = EncapsulateFunction(gpu_atomic_sum_homo_f64_i32);
//        dict["gpu_atomic_sum_homo_f64_i64"] = EncapsulateFunction(gpu_atomic_sum_homo_f64_i64);
        // heterogeneous atomic_sum
        dict["gpu_atomic_sum_heter_f32_i32"] = EncapsulateFunction(gpu_atomic_sum_heter_f32_i32);
        dict["gpu_atomic_sum_heter_f32_i64"] = EncapsulateFunction(gpu_atomic_sum_heter_f32_i64);
//        dict["gpu_atomic_sum_heter_f64_i32"] = EncapsulateFunction(gpu_atomic_sum_heter_f64_i32);
//        dict["gpu_atomic_sum_heter_f64_i64"] = EncapsulateFunction(gpu_atomic_sum_heter_f64_i64);

        // homogeneous atomic_prod
        dict["gpu_atomic_prod_homo_f32_i32"] = EncapsulateFunction(gpu_atomic_prod_homo_f32_i32);
        dict["gpu_atomic_prod_homo_f32_i64"] = EncapsulateFunction(gpu_atomic_prod_homo_f32_i64);
//        dict["gpu_atomic_prod_homo_f64_i32"] = EncapsulateFunction(gpu_atomic_prod_homo_f64_i32);
//        dict["gpu_atomic_prod_homo_f64_i64"] = EncapsulateFunction(gpu_atomic_prod_homo_f64_i64);
        // heterogeneous atomic_prod
        dict["gpu_atomic_prod_heter_f32_i32"] = EncapsulateFunction(gpu_atomic_prod_heter_f32_i32);
        dict["gpu_atomic_prod_heter_f32_i64"] = EncapsulateFunction(gpu_atomic_prod_heter_f32_i64);
//        dict["gpu_atomic_prod_heter_f64_i32"] = EncapsulateFunction(gpu_atomic_prod_heter_f64_i32);
//        dict["gpu_atomic_prod_heter_f64_i64"] = EncapsulateFunction(gpu_atomic_prod_heter_f64_i64);

        // matmul operators
        dict["vector_matmul_mask"] = EncapsulateFunction(vector_matmul_mask);
        dict["masked_matmul"] = EncapsulateFunction(masked_matmul);
        dict["uniform"] = EncapsulateFunction(uniform);
        dict["uniform2"] = EncapsulateFunction(uniform2);

        dict["mmm_8K_8x256x512"] = EncapsulateFunction(mmm_8K_8x256x512);
        dict["mmm_8K_8x128x256"] = EncapsulateFunction(mmm_8K_8x128x256);
        dict["mmm_8K_16x128x512"] = EncapsulateFunction(mmm_8K_16x128x512);
        dict["mmm_8K_16x64x256"] = EncapsulateFunction(mmm_8K_16x64x256);
        dict["mmm_8K_32x64x512"] = EncapsulateFunction(mmm_8K_32x64x512);
        dict["mmm_8K_32x32x256"] = EncapsulateFunction(mmm_8K_32x32x256);
        dict["event_mmm_8K_16x64x256"] = EncapsulateFunction(event_mmm_8K_16x64x256);


        return dict;
    }

    PYBIND11_MODULE(
            gpu_ops, m
    ) {
    m.def("registrations", &Registrations);
    m.def("build_event_sum_descriptor", &build_event_sum_descriptor);
    m.def("build_event_sum2_descriptor", &build_event_sum2_descriptor);
    m.def("build_atomic_sum_descriptor", &build_atomic_sum_descriptor);
    m.def("build_atomic_prod_descriptor", &build_atomic_prod_descriptor);
    m.def("build_event_sum_descriptor", &build_event_sum_descriptor);
    m.def("build_matmul_descriptor", &build_matmul_descriptor);
    m.def("build_rand_sample_descriptor", &build_rand_sample_descriptor);
}
}  // namespace
