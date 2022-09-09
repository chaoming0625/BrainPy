// This file contains the GPU implementation of our op. It's a pretty typical CUDA kernel
// and I make no promises about the quality of the code or the choices made therein, but
// it should get the point across.

#include "mat_mtp_mask_gpu.h"

namespace brainpy_lib {

    namespace {


        __device__ __noinline__
        bool conn_by_fixed_prob(curandState *state, int source_id, int target_id, float p) {
            return curand_uniform(state) < p;
        }


        /*
         * The operator processing "a real-value matrix" multiply "a mask matrix":
         *    R @ Mask
         *
         * Assumptions:
         *   1. K_TILE * M_TILE <= 2K
         *   2. K_TILE * M_TILE == N_TILE * 4
         */
        template<const int K_TILE,
                const int M_TILE,
                const int N_THREAD,
                const int SM_K_STRIDE>
        __global__ void mmm_8K_sm_kernel(
                const float *A,
                float *O,
                const uint32_t m,
                const uint32_t n,
                const uint32_t k,
                const curandState *rng_states, // random states
                const float prob // probability
        ) {
            // shared memory
            __shared__ __align__
            (16 * 1024)
            char smem[16 * 1024];
            float *A_smem = reinterpret_cast<float *>(smem);

            // A register fragment
            float A_frag[2][K_TILE];

            // O register fragment
            float O_frag[K_TILE][N_THREAD];
#pragma unroll
            for (int i = 0; i < K_TILE; ++i) {
#pragma unroll
                for (int j = 0; j < N_THREAD; ++j) {
                    O_frag[i][j] = 0;
                }
            }

            // start position of m axis
            uint32_t m_id_start = 0;

            // start position of K axis
            const uint32_t k_id_start = blockIdx.y * K_TILE;

            const uint32_t n_tid = threadIdx.x + blockIdx.x * blockDim.x; // 第几个线程
            // position of n
            const uint32_t n_id = n_tid * N_THREAD;

            // number of threads to read A on axis y
            const uint32_t y_id = threadIdx.x / M_TILE * 4 + k_id_start;

            // number of m tile to read A
            uint32_t num_m_tile = (m + M_TILE - 1) / M_TILE - 1;

            // random state
            curandState state = rng_states[n_tid];

            // A load register
            float A_ldg_reg[4];
            // A_tile ldg pointer
            const char *A_ldg_ptr = (const char *) (A + y_id * m + threadIdx.x % M_TILE);

            // A_tile sts/lds pointer
            // using uint32_t pointer for faster double buffer switch
            uint32_t A_sts_addr = smem_u32addr(A_smem +
                                               (threadIdx.x % M_TILE) * SM_K_STRIDE +
                                               threadIdx.x / M_TILE * 4);
            uint32_t A_lds_addr = smem_u32addr(A_smem);

            /*
             * 1'st A tile loaded before the k_tile loop
             */
            // ldg_guard to avoid LDG out of bound
            uint32_t A_ldg_guard = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if ((y_id + i) < m) { A_ldg_guard |= (1u << i); }
            }

            // load 1'st tile to shared memory
            {
                uint32_t first_m_tile = m - num_m_tile * M_TILE;

                // load a float4 of A into reg
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    bool guard = (A_ldg_guard & (1u << i)) != 0 && (threadIdx.x % M_TILE < first_m_tile);
                    ldg32_nc_0(A_ldg_reg[i], A_ldg_ptr + i * m * sizeof(float), guard);
                }

                // store a float into A_sts
                sts128(A_ldg_reg[0],
                       A_ldg_reg[1],
                       A_ldg_reg[2],
                       A_ldg_reg[3],
                       A_sts_addr);
                __syncthreads();

                // switch double buffer
                A_sts_addr ^= 0x2000;  // 0x十六进制，8K

                // ldg pointer for next tile
                A_ldg_ptr += first_m_tile * sizeof(float);  // 从左往右移动k
            }

            // load 1'st fragment
            if (n_id < n) {

#pragma unroll
                for (int i = 0; i < K_TILE; i += 4) {
                    lds128(A_frag[0][i],
                           A_frag[0][i + 1],
                           A_frag[0][i + 2],
                           A_frag[0][i + 3],
                           A_lds_addr + i * sizeof(float));
                }
            }

            /*
             * num_m_tile loop
             */
            for (; num_m_tile > 0; --num_m_tile) {
#pragma unroll
                for (int m_frag = 0; m_frag < M_TILE; ++m_frag) {
                    // store next A tile to shared memory
                    if (m_frag == M_TILE - 1) {
                        sts128(A_ldg_reg[0],
                               A_ldg_reg[1],
                               A_ldg_reg[2],
                               A_ldg_reg[3],
                               A_sts_addr);
                        __syncthreads();

                        // switch double buffer
                        A_lds_addr ^= 0x2000;
                        A_sts_addr ^= 0x2000;

                        // ldg pointer for next tile
                        A_ldg_ptr += M_TILE * sizeof(float);
                    }

                    if (n_id < n) {

                        // load next A fragment from shared memory to register
#pragma unroll
                        for (int i = 0; i < K_TILE; i += 4) {
                            lds128(A_frag[(m_frag + 1) % 2][i],
                                   A_frag[(m_frag + 1) % 2][i + 1],
                                   A_frag[(m_frag + 1) % 2][i + 2],
                                   A_frag[(m_frag + 1) % 2][i + 3],
                                   A_lds_addr + ((m_frag + 1) % M_TILE * SM_K_STRIDE + i) * sizeof(float));
                        }
                    }

                    // load next A tile
                    if (m_frag == 0) {

#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            ldg32_nc(A_ldg_reg[i],
                                     A_ldg_ptr + i * m * sizeof(float),
                                     (A_ldg_guard & (1u << i)) != 0);
                        }

                    }

                    if (n_id < n) {
                        // FFMA loop
#pragma unroll
                        for (int j = 0; j < N_THREAD; ++j) {
                            bool conn = conn_by_fixed_prob(&state, m_id_start + m_frag, n_id + j, prob);
                            if (conn) {
#pragma unroll
                                for (int i = 0; i < K_TILE; ++i) {
                                    O_frag[i][j] += A_frag[m_frag % 2][i];
                                }
                            }
                        }
                    }
                }
                m_id_start += M_TILE;
            }


            if (n_id < n) {

                // FFMA for the last tile
#pragma unroll
                for (int m_frag = 0; m_frag < M_TILE; ++m_frag) {
                    if (m_frag < M_TILE - 1) {
                        // load next A fragment from shared memory to register
#pragma unroll
                        for (int i = 0; i < K_TILE; i += 4) {
                            lds128(A_frag[(m_frag + 1) % 2][i],
                                   A_frag[(m_frag + 1) % 2][i + 1],
                                   A_frag[(m_frag + 1) % 2][i + 2],
                                   A_frag[(m_frag + 1) % 2][i + 3],
                                   A_lds_addr + ((m_frag + 1) % M_TILE * SM_K_STRIDE + i) * sizeof(float));
                        }
                    }

                    // FFMA loop
#pragma unroll
                    for (int j = 0; k < N_THREAD; ++j) {
                        bool conn = conn_by_fixed_prob(&state, m_id_start + m_frag, n_id + j, prob);
                        if (conn) {
#pragma unroll
                            for (int i = 0; i < K_TILE; ++i) {
                                O_frag[i][j] += A_frag[m_frag % 2][i];
                            }
                        }
                    }

                }

            }


            // O_tile write back
#pragma unroll
            for (int j = 0; j < N_THREAD; ++j) {
#pragma unroll
                for (int i = 0; i < K_TILE; ++i) {
                    if ((k_id_start + i < k) && (n_id + j < n)) {
                        O[(k_id_start + i) * n + n_id + j] = O_frag[i][j];
                    }
                }
            }

        }

        /*
         * The operator processing "a real-value matrix" multiply "a mask matrix":
         *    R @ Mask
         *
         * Assumptions:
         *   1. K_TILE * M_TILE <= 1K
         *   2. K_TILE * M_TILE == N_TILE * 4
         */
        template<const int K_TILE,
                const int M_TILE,
                const int N_THREAD,
                const int SM_K_STRIDE>
        __global__ void mmm_4K_sm_kernel(
                const float *A,
                float *O,
                const uint32_t m,
                const uint32_t n,
                const uint32_t k,
                const curandState *rng_states, // random states
                const float prob // probability
        ) {
            // shared memory
            __shared__ __align__
            (8 * 1024)
            char smem[8 * 1024];
            float *A_smem = reinterpret_cast<float *>(smem);

            // A register fragment
            float A_frag[2][K_TILE];

            // O register fragment
            float O_frag[K_TILE][N_THREAD];
#pragma unroll
            for (int i = 0; i < K_TILE; ++i) {
#pragma unroll
                for (int j = 0; j < N_THREAD; ++j) {
                    O_frag[i][j] = 0;
                }
            }

            // start position of m axis
            uint32_t m_id_start = 0;

            // start position of K axis
            const uint32_t k_id_start = blockIdx.y * K_TILE;

            const uint32_t n_tid = threadIdx.x + blockIdx.x * blockDim.x; // 第几个线程
            // position of n
            const uint32_t n_id = n_tid * N_THREAD;

            // number of threads to read A on axis y
            const uint32_t y_id = threadIdx.x / M_TILE * 4 + k_id_start;

            // number of m tile to read A
            uint32_t num_m_tile = (m + M_TILE - 1) / M_TILE - 1;

            // random state
            curandState state = rng_states[n_tid];

            // A load register
            float A_ldg_reg[4];
            // A_tile ldg pointer
            const char *A_ldg_ptr = (const char *) (A + y_id * m + threadIdx.x % M_TILE);

            // A_tile sts/lds pointer
            // using uint32_t pointer for faster double buffer switch
            uint32_t A_sts_addr = smem_u32addr(A_smem +
                                               (threadIdx.x % M_TILE) * SM_K_STRIDE +
                                               threadIdx.x / M_TILE * 4);
            uint32_t A_lds_addr = smem_u32addr(A_smem);

            /*
             * 1'st A tile loaded before the k_tile loop
             */
            // ldg_guard to avoid LDG out of bound
            uint32_t A_ldg_guard = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if ((y_id + i) < m) { A_ldg_guard |= (1u << i); }
            }

            // load 1'st tile to shared memory
            {
                uint32_t first_m_tile = m - num_m_tile * M_TILE;

                // load a float4 of A into reg
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    bool guard = (A_ldg_guard & (1u << i)) != 0 && (threadIdx.x % M_TILE < first_m_tile);
                    ldg32_nc_0(A_ldg_reg[i], A_ldg_ptr + i * m * sizeof(float), guard);
                }

                // store a float into A_sts
                sts128(A_ldg_reg[0],
                       A_ldg_reg[1],
                       A_ldg_reg[2],
                       A_ldg_reg[3],
                       A_sts_addr);
                __syncthreads();

                // switch double buffer
                A_sts_addr ^= 0x1000;  // 0x十六进制，8K

                // ldg pointer for next tile
                A_ldg_ptr += first_m_tile * sizeof(float);  // 从左往右移动k
            }

            // load 1'st fragment
            if (n_id < n) {

#pragma unroll
                for (int i = 0; i < K_TILE; i += 4) {
                    lds128(A_frag[0][i],
                           A_frag[0][i + 1],
                           A_frag[0][i + 2],
                           A_frag[0][i + 3],
                           A_lds_addr + i * sizeof(float));
                }
            }

            /*
             * num_m_tile loop
             */
            for (; num_m_tile > 0; --num_m_tile) {
#pragma unroll
                for (int m_frag = 0; m_frag < M_TILE; ++m_frag) {
                    // store next A tile to shared memory
                    if (m_frag == M_TILE - 1) {
                        sts128(A_ldg_reg[0],
                               A_ldg_reg[1],
                               A_ldg_reg[2],
                               A_ldg_reg[3],
                               A_sts_addr);
                        __syncthreads();

                        // switch double buffer
                        A_lds_addr ^= 0x1000;
                        A_sts_addr ^= 0x1000;

                        // ldg pointer for next tile
                        A_ldg_ptr += M_TILE * sizeof(float);
                    }

                    if (n_id < n) {

                        // load next A fragment from shared memory to register
#pragma unroll
                        for (int i = 0; i < K_TILE; i += 4) {
                            lds128(A_frag[(m_frag + 1) % 2][i],
                                   A_frag[(m_frag + 1) % 2][i + 1],
                                   A_frag[(m_frag + 1) % 2][i + 2],
                                   A_frag[(m_frag + 1) % 2][i + 3],
                                   A_lds_addr + ((m_frag + 1) % M_TILE * SM_K_STRIDE + i) * sizeof(float));
                        }
                    }

                    // load next A tile
                    if (m_frag == 0) {
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            ldg32_nc(A_ldg_reg[i],
                                     A_ldg_ptr + i * m * sizeof(float),
                                     (A_ldg_guard & (1u << i)) != 0);
                        }
                    }

                    if (n_id < n) {
                        // FFMA loop
#pragma unroll
                        for (int j = 0; j < N_THREAD; ++j) {
                            bool conn = conn_by_fixed_prob(&state, m_id_start + m_frag, n_id + j, prob);
                            if (conn) {
#pragma unroll
                                for (int i = 0; i < K_TILE; ++i) {
                                    O_frag[i][j] += A_frag[m_frag % 2][i];
                                }
                            }
                        }
                    }
                }
                m_id_start += M_TILE;
            }


            if (n_id < n) {

                // FFMA for the last tile
#pragma unroll
                for (int m_frag = 0; m_frag < M_TILE; ++m_frag) {
                    if (m_frag < M_TILE - 1) {
                        // load next A fragment from shared memory to register
#pragma unroll
                        for (int i = 0; i < K_TILE; i += 4) {
                            lds128(A_frag[(m_frag + 1) % 2][i],
                                   A_frag[(m_frag + 1) % 2][i + 1],
                                   A_frag[(m_frag + 1) % 2][i + 2],
                                   A_frag[(m_frag + 1) % 2][i + 3],
                                   A_lds_addr + ((m_frag + 1) % M_TILE * SM_K_STRIDE + i) * sizeof(float));
                        }
                    }

                    // FFMA loop
#pragma unroll
                    for (int j = 0; k < N_THREAD; ++j) {
                        bool conn = conn_by_fixed_prob(&state, m_id_start + m_frag, n_id + j, prob);
                        if (conn) {
#pragma unroll
                            for (int i = 0; i < K_TILE; ++i) {
                                O_frag[i][j] += A_frag[m_frag % 2][i];
                            }
                        }
                    }

                }

            }


            // O_tile write back
#pragma unroll
            for (int j = 0; j < N_THREAD; ++j) {
#pragma unroll
                for (int i = 0; i < K_TILE; ++i) {
                    if ((k_id_start + i < k) && (n_id + j < n)) {
                        O[(k_id_start + i) * n + n_id + j] = O_frag[i][j];
                    }
                }
            }

        }


    }  // namespace


//    void mmm_8K_1x8x256x512(
//            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
//    ) {
//        // size
//        const MMMDescriptor &d = *UnpackDescriptor<MMMDescriptor>(opaque, opaque_len);
//        const std::uint32_t m = d.m;
//        const std::uint32_t k = d.k;
//        const std::uint32_t n = d.n;
//        const float p = d.p;
//
//        // input and output data
//        const float *A = reinterpret_cast<const float *>(buffers[0]);
//        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[1]);
//        float *O = reinterpret_cast<float *>(buffers[2]);
//
//        // call kernel
//        cudaMemset(O, 0, sizeof(float) * n * k);
//        dim3 grid((n + 511) / 512, (k + 7) / 8);
//        mmm_8K_sm_kernel<8, 256, 1, 8><<<grid, 512, 0, stream>>>(A, O, m, n, k, rng_states, p);
//        ThrowIfError(cudaGetLastError());
//    }


    void mmm_8K_1x8x128x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        // size
        const MMMDescriptor &d = *UnpackDescriptor<MMMDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const float p = d.p;

        // input and output data
        const float *A = reinterpret_cast<const float *>(buffers[0]);
        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        // call kernel
        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 255) / 256, (k + 7) / 8);
        mmm_8K_sm_kernel<8, 128, 1, 12><<<grid, 256, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }


//    void mmm_8K_1x16x128x512(
//            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
//    ) {
//        // size
//        const MMMDescriptor &d = *UnpackDescriptor<MMMDescriptor>(opaque, opaque_len);
//        const std::uint32_t m = d.m;
//        const std::uint32_t k = d.k;
//        const std::uint32_t n = d.n;
//        const float p = d.p;
//
//        // input and output data
//        const float *A = reinterpret_cast<const float *>(buffers[0]);
//        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[1]);
//        float *O = reinterpret_cast<float *>(buffers[2]);
//
//        // call kernel
//        cudaMemset(O, 0, sizeof(float) * n * k);
//        dim3 grid((n + 511) / 512, (k + 15) / 16);
//        mmm_8K_sm_kernel<16, 128, 1, 16><<<grid, 512, 0, stream>>>(A, O, m, n, k, rng_states, p);
//        ThrowIfError(cudaGetLastError());
//    }


    void mmm_8K_1x16x64x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        // size
        const MMMDescriptor &d = *UnpackDescriptor<MMMDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const float p = d.p;

        // input and output data
        const float *A = reinterpret_cast<const float *>(buffers[0]);
        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        // call kernel
        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 256) / 256, (k + 15) / 16);
        mmm_8K_sm_kernel<16, 64, 1, 20><<<grid, 256, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }


//    void mmm_8K_1x32x64x512(
//            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
//    ) {
//        // size
//        const MMMDescriptor &d = *UnpackDescriptor<MMMDescriptor>(opaque, opaque_len);
//        const std::uint32_t m = d.m;
//        const std::uint32_t k = d.k;
//        const std::uint32_t n = d.n;
//        const float p = d.p;
//
//        // input and output data
//        const float *A = reinterpret_cast<const float *>(buffers[0]);
//        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[1]);
//        float *O = reinterpret_cast<float *>(buffers[2]);
//
//        // call kernel
//        cudaMemset(O, 0, sizeof(float) * n * k);
//        dim3 grid((n + 511) / 512, (k + 31) / 32);
//        mmm_8K_sm_kernel<32, 64, 1, 32><<<grid, 512, 0, stream>>>(A, O, m, n, k, rng_states, p);
//        ThrowIfError(cudaGetLastError());
//    }


    void mmm_8K_1x32x32x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        // size
        const MMMDescriptor &d = *UnpackDescriptor<MMMDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const float p = d.p;

        // input and output data
        const float *A = reinterpret_cast<const float *>(buffers[0]);
        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        // call kernel
        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 255) / 256, (k + 31) / 32);
        mmm_8K_sm_kernel<32, 32, 1, 36><<<grid, 256, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }



    void mmm_8K_1x64x16x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        // size
        const MMMDescriptor &d = *UnpackDescriptor<MMMDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const float p = d.p;

        // input and output data
        const float *A = reinterpret_cast<const float *>(buffers[0]);
        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        // call kernel
        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 255) / 256, (k + 31) / 32);
        mmm_8K_sm_kernel<64, 16, 1, 68><<<grid, 256, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }


    void mmm_8K_4x8x128x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        // size
        const MMMDescriptor &d = *UnpackDescriptor<MMMDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const float p = d.p;

        // input and output data
        const float *A = reinterpret_cast<const float *>(buffers[0]);
        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        // call kernel
        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 1023) / 1024, (k + 7) / 8);
        mmm_8K_sm_kernel<8, 128, 1, 12><<<grid, 256, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }


    void mmm_8K_4x16x64x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        // size
        const MMMDescriptor &d = *UnpackDescriptor<MMMDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const float p = d.p;

        // input and output data
        const float *A = reinterpret_cast<const float *>(buffers[0]);
        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        // call kernel
        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 1023) / 1024, (k + 15) / 16);
        mmm_8K_sm_kernel<16, 64, 4, 20><<<grid, 256, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }


    void mmm_8K_4x32x32x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        // size
        const MMMDescriptor &d = *UnpackDescriptor<MMMDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const float p = d.p;

        // input and output data
        const float *A = reinterpret_cast<const float *>(buffers[0]);
        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        // call kernel
        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 1023) / 1024, (k + 31) / 32);
        mmm_8K_sm_kernel<32, 32, 4, 36><<<grid, 256, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }



    void mmm_8K_4x64x16x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        // size
        const MMMDescriptor &d = *UnpackDescriptor<MMMDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const float p = d.p;

        // input and output data
        const float *A = reinterpret_cast<const float *>(buffers[0]);
        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        // call kernel
        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 1023) / 1024, (k + 31) / 32);
        mmm_8K_sm_kernel<64, 16, 4, 68><<<grid, 256, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }


    void mmm_4K_1x8x128x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        // size
        const MMMDescriptor &d = *UnpackDescriptor<MMMDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const float p = d.p;

        // input and output data
        const float *A = reinterpret_cast<const float *>(buffers[0]);
        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        // call kernel
        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 255) / 256, (k + 7) / 8);
        mmm_4K_sm_kernel<8, 128, 1, 8><<<grid, 256, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }


    void mmm_4K_1x16x64x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        // size
        const MMMDescriptor &d = *UnpackDescriptor<MMMDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const float p = d.p;

        // input and output data
        const float *A = reinterpret_cast<const float *>(buffers[0]);
        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        // call kernel
        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 255) / 256, (k + 15) / 16);
        mmm_4K_sm_kernel<16, 64, 1, 16><<<grid, 256, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }

    void mmm_4K_1x32x32x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        // size
        const MMMDescriptor &d = *UnpackDescriptor<MMMDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const float p = d.p;

        // input and output data
        const float *A = reinterpret_cast<const float *>(buffers[0]);
        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        // call kernel
        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 255) / 256, (k + 31) / 32);
        mmm_8K_sm_kernel<32, 32, 1, 32><<<grid, 256, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }


    void mmm_4K_1x64x16x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        // size
        const MMMDescriptor &d = *UnpackDescriptor<MMMDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const float p = d.p;

        // input and output data
        const float *A = reinterpret_cast<const float *>(buffers[0]);
        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        // call kernel
        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 255) / 256, (k + 63) / 64);
        mmm_8K_sm_kernel<64, 16, 1, 64><<<grid, 256, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }


    void mmm_4K_4x8x128x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        // size
        const MMMDescriptor &d = *UnpackDescriptor<MMMDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const float p = d.p;

        // input and output data
        const float *A = reinterpret_cast<const float *>(buffers[0]);
        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        // call kernel
        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 1023) / 1024, (k + 7) / 8);
        mmm_4K_sm_kernel<8, 128, 4, 8><<<grid, 256, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }


    void mmm_4K_4x16x64x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        // size
        const MMMDescriptor &d = *UnpackDescriptor<MMMDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const float p = d.p;

        // input and output data
        const float *A = reinterpret_cast<const float *>(buffers[0]);
        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        // call kernel
        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 1023) / 1024, (k + 15) / 16);
        mmm_4K_sm_kernel<16, 64, 1, 16><<<grid, 256, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }


    void mmm_4K_4x32x32x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        // size
        const MMMDescriptor &d = *UnpackDescriptor<MMMDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const float p = d.p;

        // input and output data
        const float *A = reinterpret_cast<const float *>(buffers[0]);
        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        // call kernel
        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 1023) / 1024, (k + 31) / 32);
        mmm_8K_sm_kernel<32, 32, 1, 32><<<grid, 256, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }


    void mmm_4K_4x64x16x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        // size
        const MMMDescriptor &d = *UnpackDescriptor<MMMDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const float p = d.p;

        // input and output data
        const float *A = reinterpret_cast<const float *>(buffers[0]);
        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        // call kernel
        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 1023) / 1024, (k + 63) / 64);
        mmm_8K_sm_kernel<64, 16, 4, 64><<<grid, 256, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }




}  // namespace brainpylib
