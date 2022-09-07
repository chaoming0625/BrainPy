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

        // The operator processing "a real-value matrix" multiply "a mask matrix":
        //    R @ Mask
        //
        // Assumptions:
        // 1. K_TILE * M_TILE <= 2 * 2^10, like 16 * 128
        // 2. K_TILE * M_TILE == N_TILE
        template<
                const int K_TILE, // height of the matrix A
                const int M_TILE // width of the matrix A
        >
        __global__ void mat_mul_mask_kernel(
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
            float O_frag[K_TILE];
#pragma unroll
            for (int i = 0; i < K_TILE; ++i) { O_frag[i] = 0; }

            // position of n
            const uint32_t n_id = threadIdx.x + blockIdx.x * blockDim.x; // 第几个线程
            // position of K
            const uint32_t k_id_start = blockIdx.y * K_TILE;
            // position of m
            uint32_t m_id_start = 0;

            // A load register
            float A_ldg_reg;
            // A_tile ldg pointer
            const char *A_ldg_ptr = (const char *) (A + (threadIdx.x / M_TILE + k_id_start) * m + threadIdx.x % M_TILE);

            // random state
            curandState state = rng_states[n_id];

            // A_tile sts/lds pointer
            // using uint32_t pointer for faster double buffer switch
            uint32_t A_sts_addr = smem_u32addr(A_smem + (threadIdx.x % M_TILE) * K_TILE + threadIdx.x / M_TILE);
            uint32_t A_lds_addr = smem_u32addr(A_smem);

            /*
             * 1'st A tile loaded before the k_tile loop
             */
            uint32_t num_m_tile = (m + M_TILE - 1) / M_TILE - 1;

            // ldg_guard to avoid LDG out of bound
            uint32_t A_ldg_guard = 0;
            if (k_id_start + threadIdx.x / M_TILE < k) { A_ldg_guard = 1u; }

            // load 1'st tile to shared memory
            {
                uint32_t first_m_tile = m - num_m_tile * M_TILE;

                // load a float4 of A into reg
                bool guard = (A_ldg_guard != 0) && (threadIdx.x % M_TILE < first_m_tile);
                ldg32_nc_0(A_ldg_reg, A_ldg_ptr, guard);

                // store a float into A_sts
                sts32(A_ldg_reg, A_sts_addr);
                __syncthreads();

                // switch double buffer
                A_sts_addr ^= 0x2000;  // 0x十六进制，8K

                // ldg pointer for next tile
                A_ldg_ptr += first_m_tile * sizeof(float);  // 从左往右移动k
            }

            // load 1'st fragment
            if (n_id < n) {


#pragma unroll
                for (int i = 0; i < K_TILE / 4; ++i) {
                    lds128(A_frag[0][i],
                           A_frag[0][i + 1],
                           A_frag[0][i + 2],
                           A_frag[0][i + 3],
                           A_lds_addr + 4 * sizeof(float) * i);
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
                        sts32(A_ldg_reg, A_sts_addr);
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
                        for (int i = 0; i < K_TILE / 4; ++i) {
                            lds128(A_frag[(m_frag + 1) % 2][i],
                                   A_frag[(m_frag + 1) % 2][i + 1],
                                   A_frag[(m_frag + 1) % 2][i + 2],
                                   A_frag[(m_frag + 1) % 2][i + 3],
                                   A_lds_addr + ((m_frag + 1) * K_TILE + 4 * i) * sizeof(float));
                        }


                        // load next A tile
                        if (m_frag == 0) {
                            ldg32_nc(A_ldg_reg, A_ldg_ptr, A_ldg_guard != 0);
                        }

                        // FFMA loop
                        bool conn = conn_by_fixed_prob(&state, m_id_start + m_frag, n_id, prob);
                        if (conn) {
#pragma unroll
                            for (int i = 0; i < K_TILE; ++i) {
                                O_frag[i] += A_frag[m_frag % 2][i];
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
                        for (int i = 0; i < K_TILE / 4; ++i) {
                            lds128(A_frag[(m_frag + 1) % 2][i],
                                   A_frag[(m_frag + 1) % 2][i + 1],
                                   A_frag[(m_frag + 1) % 2][i + 2],
                                   A_frag[(m_frag + 1) % 2][i + 3],
                                   A_lds_addr + ((m_frag + 1) * K_TILE + 4 * i) * sizeof(float));
                        }
                    }

                    // FFMA loop
                    bool conn = conn_by_fixed_prob(&state, m_id_start + m_frag, n_id, prob);
                    if (conn) {
#pragma unroll
                        for (int i = 0; i < K_TILE; ++i) {
                            O_frag[i] += A_frag[m_frag % 2][i];
                        }
                    }
                }

            }


            // O_tile write back
            if (n >= n_id) {
                return;
            } else {
                for (int i = 0; i < K_TILE; ++i) {
                    if (k_id_start + i < k) {
                        O[(k_id_start + i) * n + n_id] = O_frag[i];
                    }
                }
            }
        }

        // The operator processing "a real-value matrix" multiply "a mask matrix":
        //    R @ Mask
        //
        // Assumptions:
        // 1. K_TILE * M_TILE <= 2K
        // 2. K_TILE * M_TILE == N_TILE * 4
        template<const int K_TILE, const int M_TILE, const int SM_K_STRIDE>
        __global__ void mat_mtp_mask_8K_sm_kernel(
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
            float O_frag[K_TILE];
#pragma unroll
            for (int i = 0; i < K_TILE; ++i) { O_frag[i] = 0; }

            // start position of m axis
            uint32_t m_id_start = 0;

            // start position of K axis
            const uint32_t k_id_start = blockIdx.y * K_TILE;

            // position of n
            const uint32_t n_id = threadIdx.x + blockIdx.x * blockDim.x; // 第几个线程

            // number of threads to read A on axis y
            const uint32_t y_id = threadIdx.x / M_TILE * 4 + k_id_start;

            // number of m tile to read A
            uint32_t num_m_tile = (m + M_TILE - 1) / M_TILE - 1;

            // random state
            curandState state = rng_states[n_id];

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


                        // load next A tile
                        if (m_frag == 0) {

#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                ldg32_nc(A_ldg_reg[i],
                                         A_ldg_ptr + i * m * sizeof(float),
                                         (A_ldg_guard & (1u << i)) != 0);
                            }

                        }

                        // FFMA loop
                        bool conn = conn_by_fixed_prob(&state, m_id_start + m_frag, n_id, prob);
                        if (conn) {
#pragma unroll
                            for (int i = 0; i < K_TILE; ++i) {
                                O_frag[i] += A_frag[m_frag % 2][i];
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
                    bool conn = conn_by_fixed_prob(&state, m_id_start + m_frag, n_id, prob);
                    if (conn) {
#pragma unroll
                        for (int i = 0; i < K_TILE; ++i) {
                            O_frag[i] += A_frag[m_frag % 2][i];
                        }
                    }
                }

            }


            // O_tile write back
            if (n_id < n) {
                for (int i = 0; i < K_TILE; ++i) {
                    if (k_id_start + i < k) {
                        O[(k_id_start + i) * n + n_id] = O_frag[i];
                    }
                }
            }
        }


        // The operator processing "a real-value matrix" multiply "a mask matrix":
        //    R @ Mask
        //
        // Assumptions:
        // 1. K_TILE * M_TILE <= 2K
        // 2. K_TILE * M_TILE == N_TILE * 4 * N_THREAD
        template<const int K_TILE, const int M_TILE, const int SM_K_STRIDE, const int N_THREAD>
        __global__ void event_mmm_sm_kernel(
                const curandState *rng_states,
                const bool *V,
                const float *A,
                const uint32_t m,
                const uint32_t n,
                const uint32_t k,
                const float prob // probability
                float *O,
        ) {
            // shared memory
            __shared__ char smem[8 * K_TILE * M_TILE];
            float *A_smem = reinterpret_cast<float *>(smem);
            __shared__ bool V_smem[2][M_TILE];

            // V register fragment
            bool V_frag[2];

            // A register fragment
            float A_frag[2][K_TILE];

            // O register fragment
            float O_frag[K_TILE * N_THREAD];
#pragma unroll
            for (int i = 0; i < K_TILE; ++i) {
                for (int j = 0; j < N_THREAD; ++j) {
                    O_frag[i * N_THREAD + j] = 0;
                }
            }

            // start position of K axis
            const uint32_t k_id_start = blockIdx.y * K_TILE;

            // position of n
            const uint32_t n_tid = threadIdx.x + blockIdx.x * blockDim.x;
            const uint32_t n_id = n_tid * N_THREAD;

            // number of threads to read A on axis y
            const uint32_t y_id = threadIdx.x / M_TILE * 4 + k_id_start;

            // V load register
            bool V_ldg_reg;
            // V_tile ldg pointer
//            const char *V_ldg_ptr = (const char *) (V + threadIdx.x);
            // A load register
            float A_ldg_reg[4];
            // A_tile ldg pointer
            const char *A_ldg_ptr = (const char *) (A + y_id * m + threadIdx.x % M_TILE);

            // start position of m axis
            uint32_t m_id_start = 0;

            // number of m tile to read A
            uint32_t num_m_tile = (m + M_TILE - 1) / M_TILE - 1;

            // random state
            curandState state = rng_states[n_tid];


            // ldg_guard to avoid LDG out of bound
            uint32_t A_ldg_guard = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if ((y_id + i) < m) { A_ldg_guard |= (1u << i); }
            }
            // A_tile sts/lds pointer
            uint32_t A_sts_addr = smem_u32addr(A_smem +
                                               (threadIdx.x % M_TILE) * SM_K_STRIDE +
                                               threadIdx.x / M_TILE * 4);
            uint32_t A_lds_addr = smem_u32addr(A_smem);


            uint32_t first_m_tile = m - num_m_tile * M_TILE;
            uint32_t V_ldg_idx = threadIdx.x;
            uint32_t V_lds_idx = 0;

            // load 1'st V tile to shared memory //
            if (threadIdx.x < M_TILE) {
                V_ldg_reg = threadIdx.x < first_m_tile ? V[V_ldg_idx] : false;
                V_smem[0][threadIdx.x] = V_ldg_reg;
            }
            __syncthreads();
            V_ldg_idx += first_m_tile;

            // load 1'st A tile //
            V_frag = V_smem[0][threadIdx.x % M_TILE];
            if (V_frag) {
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
            }
            __syncthreads();
            // switch double buffer
            A_sts_addr ^= 0x2000;  // 0x十六进制，8K
            // ldg pointer for next tile
            A_ldg_ptr += first_m_tile * sizeof(float);  // 从左往右移动k


            // load 1'st fragment
#pragma unroll
            for (int i = 0; i < K_TILE; i += 4) {
                lds128(A_frag[0][i],
                       A_frag[0][i + 1],
                       A_frag[0][i + 2],
                       A_frag[0][i + 3],
                       A_lds_addr + i * sizeof(float));
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


                        // load next A tile
                        if (m_frag == 0) {

#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                ldg32_nc(A_ldg_reg[i],
                                         A_ldg_ptr + i * m * sizeof(float),
                                         (A_ldg_guard & (1u << i)) != 0);
                            }

                        }

                        // FFMA loop
                        bool conn = conn_by_fixed_prob(&state, m_id_start + m_frag, n_id, prob);
                        if (conn) {
#pragma unroll
                            for (int i = 0; i < K_TILE; ++i) {
                                O_frag[i] += A_frag[m_frag % 2][i];
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
                    bool conn = conn_by_fixed_prob(&state, m_id_start + m_frag, n_id, prob);
                    if (conn) {
#pragma unroll
                        for (int i = 0; i < K_TILE; ++i) {
                            O_frag[i] += A_frag[m_frag % 2][i];
                        }
                    }
                }

            }


            // O_tile write back
            if (n_id < n) {
                for (int i = 0; i < K_TILE; ++i) {
                    if (k_id_start + i < k) {
                        O[(k_id_start + i) * n + n_id] = O_frag[i];
                    }
                }
            }
        }


    }  // namespace


    void mat_mul_mask_16x128x2048(cudaStream_t stream,
                                  void **buffers,
                                  const char *opaque,
                                  std::size_t opaque_len) {
        // size
        const MatMulDescriptor &d = *UnpackDescriptor<MatMulDescriptor>(opaque, opaque_len);
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
        dim3 grid((n + 2047) / 2048, (k + 15) / 16);
        mat_mul_mask_kernel<16, 128><<<grid, 2048, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }


    void mmm_8K_8x256x512(
            cudaStream_t stream,
            void **buffers,
            const char *opaque,
            std::size_t opaque_len
    ) {
        // size
        const MatMulDescriptor &d = *UnpackDescriptor<MatMulDescriptor>(opaque, opaque_len);
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
        dim3 grid((n + 511) / 512, (k + 7) / 8);
        mat_mtp_mask_8K_sm_kernel<8, 256, 8><<<grid, 512, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }


    void mmm_8K_8x128x256(
            cudaStream_t stream,
            void **buffers,
            const char *opaque,
            std::size_t opaque_len
    ) {
        // size
        const MatMulDescriptor &d = *UnpackDescriptor<MatMulDescriptor>(opaque, opaque_len);
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
        mat_mtp_mask_8K_sm_kernel<8, 128, 12><<<grid, 256, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }


    void mmm_8K_16x128x512(
            cudaStream_t stream,
            void **buffers,
            const char *opaque,
            std::size_t opaque_len
    ) {
        // size
        const MatMulDescriptor &d = *UnpackDescriptor<MatMulDescriptor>(opaque, opaque_len);
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
        dim3 grid((n + 511) / 512, (k + 15) / 16);
        mat_mtp_mask_8K_sm_kernel<16, 128, 16><<<grid, 512, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }


    void mmm_8K_16x64x256(
            cudaStream_t stream,
            void **buffers,
            const char *opaque,
            std::size_t opaque_len
    ) {
        // size
        const MatMulDescriptor &d = *UnpackDescriptor<MatMulDescriptor>(opaque, opaque_len);
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
        mat_mtp_mask_8K_sm_kernel<16, 64, 20><<<grid, 256, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }


    void mmm_8K_32x64x512(
            cudaStream_t stream,
            void **buffers,
            const char *opaque,
            std::size_t opaque_len
    ) {
        // size
        const MatMulDescriptor &d = *UnpackDescriptor<MatMulDescriptor>(opaque, opaque_len);
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
        dim3 grid((n + 511) / 512, (k + 31) / 32);
        mat_mtp_mask_8K_sm_kernel<32, 64, 32><<<grid, 512, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }


    void mmm_8K_32x32x256(
            cudaStream_t stream,
            void **buffers,
            const char *opaque,
            std::size_t opaque_len
    ) {
        // size
        const MatMulDescriptor &d = *UnpackDescriptor<MatMulDescriptor>(opaque, opaque_len);
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
        mat_mtp_mask_8K_sm_kernel<32, 32, 36><<<grid, 256, 0, stream>>>(A, O, m, n, k, rng_states, p);
        ThrowIfError(cudaGetLastError());
    }


    void event_mmm_8K_16x64x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        // size
        const MatMulDescriptor &d = *UnpackDescriptor<MatMulDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const float p = d.p;

        // input and output data
        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[0]);
        const bool *V = reinterpret_cast<const bool *>(buffers[1]);
        const float *A = reinterpret_cast<const float *>(buffers[2]);
        float *O = reinterpret_cast<float *>(buffers[3]);

        // call kernel
        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 256) / 256, (k + 15) / 16);
        event_mmm_sm_kernel<16, 64, 20><<<grid, 256, 0, stream>>>(rng_states, V, A, m, n, k, p, O);
        ThrowIfError(cudaGetLastError());
    }


}  // namespace brainpylib
