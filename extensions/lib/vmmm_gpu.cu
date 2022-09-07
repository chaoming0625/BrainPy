// This file contains the GPU implementation of our op. It's a pretty typical CUDA kernel
// and I make no promises about the quality of the code or the choices made therein, but
// it should get the point across.

#include "vmmm_gpu.h"

namespace brainpy_lib {

    namespace {




        __device__ __noinline__
        bool conn_by_fixed_prob(curandState state, int source_id, int target_id, float p) {
            return curand_uniform(&state) < p;
        }


        __global__ __launch_bounds__(256, 2)
        void vector_matmul_mask_kernel(
                const float *V,
                const float *A,
                const float *B,
                float *O,
                const uint32_t m,
                const uint32_t n,
                const uint32_t k,
                const uint32_t A_ldg_step, // k * sizeof(float)
                const uint32_t B_ldg_step,
                const uint32_t seed, // random seed
                const float prob // probability
        ) {
            // n * sizeof(float) * 8
            __shared__ __align__
            (16 * 1024)
            char smem[24 * 1024];
            float *A_smem = reinterpret_cast<float *>(smem);
            float *B_smem = reinterpret_cast<float *>(smem + 16 * 1024);
            __shared__ char vec_smem[128 * 4];
            float *V_smem = reinterpret_cast<float *>(vec_smem);

            // V, A, B, C, and O register fragment
            float V_frag[8];
            float A_frag[2][8];
            float B_frag[2][8];
            float C_frag[8][8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {  // 初始化矩阵C: C_frag
#pragma unroll
                for (int j = 0; j < 8; ++j) { C_frag[i][j] = 0; }
            }
            float O_frag[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) { O_frag[i] = 0; } // 初始化输出output

            float V_ldg_reg;
            float A_ldg_reg[4];
            float B_ldg_reg[4];

            const uint32_t warp_id = threadIdx.x / 32; // 第几个warp
            const uint32_t lane_id = threadIdx.x % 32; // warp中的第几个线程

            // 4x8 threads each warp for FFMA
            const uint32_t mma_tid_x = (lane_id / 2) % 8;
            const uint32_t mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);

            // build connections
            uint32_t m_idx1 = blockIdx.y * 128 + warp_id / 2 * 32 + mma_tid_y;
            uint32_t n_idx1 = blockIdx.x * 128 + warp_id % 2 * 64 + mma_tid_x;
            curandState state;
//             curand_init(seed, m_idx1 * n + n_idx1, 0, &state);
            curand_init(seed + m_idx1 * n + n_idx1, 0, 0, &state);
            bool conn[8][8];
#pragma unroll
            for (int i = 0; i < 2; i++) {
#pragma unroll
                for (int j = 0; j < 2; j++) {
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        int pos_y = i * 4 + p;
                        int pos_x = j * 4;
                        conn[pos_y][pos_x] = conn_by_fixed_prob(state, m_idx1 + pos_y, n_idx1 + pos_x, prob);
                        conn[pos_y][pos_x+1] = conn_by_fixed_prob(state, m_idx1 + pos_y, n_idx1 + pos_x+1, prob);
                        conn[pos_y][pos_x+2] = conn_by_fixed_prob(state, m_idx1 + pos_y, n_idx1 + pos_x+2, prob);
                        conn[pos_y][pos_x+3] = conn_by_fixed_prob(state, m_idx1 + pos_y, n_idx1 + pos_x+3, prob);
                    }
                }
            }

            // A_tile & B_tile ldg pointer
            // 加载global memory数据时线程的分配方案
            const char *A_ldg_ptr = (const char *) (A + (blockIdx.y * 128 + threadIdx.x / 8 * 4) * k + threadIdx.x % 8);
            const char *B_ldg_ptr = (const char *) (B + (threadIdx.x / 32) * n + blockIdx.x * 128 + threadIdx.x % 32);
            const char *V_ldg_ptr = (const char *) (V + blockIdx.y * 128 + threadIdx.x / 2);

            // A_tile & B_tile sts/lds pointer
            // using uint32_t pointer for faster double buffer switch
            // 存储shared memory数据时线程的分配方案，与“加载global memory数据时线程的分配方案”相同
            uint32_t A_sts_addr = smem_u32addr(A_smem + (threadIdx.x % 8) * 132 + (threadIdx.x / 8) * 4);
            uint32_t B_sts_addr = smem_u32addr(B_smem + (threadIdx.x / 32) * 128 + (threadIdx.x % 32));
            uint32_t V_sts_addr = smem_u32addr(V_smem + threadIdx.x / 2);

            // 加载shared memory数据时线程的分配方案，与“each warp for FFMA”相同
            uint32_t A_lds_addr = smem_u32addr(A_smem + (warp_id / 2) * 32 + mma_tid_y * 4);
            uint32_t B_lds_addr = smem_u32addr(B_smem + (warp_id % 2) * 64 + mma_tid_x * 4);
            uint32_t V_lds_addr = smem_u32addr(V_smem + (warp_id / 2) * 32 + mma_tid_y * 4);

            /*
             * 1'st A&B tile loaded before the k_tile loop
             */
            uint32_t k_tiles = (k + 7) / 8 - 1;

            // ldg_guard to avoid LDG out of bound
            uint32_t A_ldg_guard = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int m_idx = blockIdx.y * 128 + threadIdx.x / 8 * 4 + i;
                if (m_idx < m) { A_ldg_guard |= (1u << i); }
            }

            uint32_t B_ldg_guard = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int n_idx = blockIdx.x * 128 + threadIdx.x % 32 + i * 32;
                if (n_idx < n) { B_ldg_guard |= (1u << i); }
            }

            // load a float of V into smem
            if (threadIdx.x % 2 == 0) {
                ldg32_nc_0(V_ldg_reg, V_ldg_ptr, (blockIdx.y * 128 + threadIdx.x / 2) < m);
                sts32(V_ldg_reg, V_sts_addr);
            }

            // load 1'st tile to shared memory
            {
                uint32_t first_k_tile = k - k_tiles * 8;

                // load a float4 of A into reg
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    bool guard = (A_ldg_guard & (1u << i)) != 0 && threadIdx.x % 8 < first_k_tile;
                    ldg32_nc_0(A_ldg_reg[i], A_ldg_ptr + i * A_ldg_step, guard);
                }
                // store a float4 into A_sts
                sts128(A_ldg_reg[0],
                       A_ldg_reg[1],
                       A_ldg_reg[2],
                       A_ldg_reg[3],
                       A_sts_addr);

                // load a float4 of B into reg
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    bool guard = (B_ldg_guard & (1u << i)) != 0 && threadIdx.x / 32 < first_k_tile;
                    ldg32_nc_0(B_ldg_reg[i], B_ldg_ptr + i * 32 * sizeof(float), guard);
                }
                // store a float into B_sts
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sts32(B_ldg_reg[i], B_sts_addr + i * 32 * sizeof(float));
                }

                __syncthreads();

                // switch double buffer
                A_sts_addr ^= 0x2000;  // 0x十六进制，8K
                B_sts_addr ^= 0x1000;  // 0x十六进制，4K

                // ldg pointer for next tile
                A_ldg_ptr += first_k_tile * sizeof(float);  // 从左往右移动k?
                B_ldg_ptr += n * first_k_tile * sizeof(float);  // 从上往下移动k?
            }

            // load V fragment
            lds128(V_frag[0], V_frag[1], V_frag[2], V_frag[3], V_lds_addr);
            lds128(V_frag[4], V_frag[5], V_frag[6], V_frag[7], V_lds_addr + 16 * sizeof(float));


            // load 1'st fragment
            lds128(A_frag[0][0],
                   A_frag[0][1],
                   A_frag[0][2],
                   A_frag[0][3],
                   A_lds_addr);
            lds128(A_frag[0][4],
                   A_frag[0][5],
                   A_frag[0][6],
                   A_frag[0][7],
                   A_lds_addr + 16 * sizeof(float));
            lds128(B_frag[0][0],
                   B_frag[0][1],
                   B_frag[0][2],
                   B_frag[0][3],
                   B_lds_addr);
            lds128(B_frag[0][4],
                   B_frag[0][5],
                   B_frag[0][6],
                   B_frag[0][7],
                   B_lds_addr + 32 * sizeof(float));


            /*
             * k_tiles loop
             */
            for (; k_tiles > 0; --k_tiles) {
#pragma unroll
                for (int k_frag = 0; k_frag < 8; ++k_frag) {
                    // store next A&B tile to shared memory
                    if (k_frag == 7) {
                        sts128(A_ldg_reg[0],
                               A_ldg_reg[1],
                               A_ldg_reg[2],
                               A_ldg_reg[3],
                               A_sts_addr);
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            sts32(B_ldg_reg[i], B_sts_addr + i * 32 * sizeof(float));
                        }

                        __syncthreads();

                        // switch double buffer
                        A_lds_addr ^= 0x2000;
                        B_lds_addr ^= 0x1000;
                        A_sts_addr ^= 0x2000;
                        B_sts_addr ^= 0x1000;

                        // ldg pointer for next tile
                        A_ldg_ptr += 8 * sizeof(float);
                        B_ldg_ptr += B_ldg_step;
                    }

                    // load next A&B fragment from shared memory to register
                    lds128(A_frag[(k_frag + 1) % 2][0],
                           A_frag[(k_frag + 1) % 2][1],
                           A_frag[(k_frag + 1) % 2][2],
                           A_frag[(k_frag + 1) % 2][3],
                           A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
                    lds128(A_frag[(k_frag + 1) % 2][4],
                           A_frag[(k_frag + 1) % 2][5],
                           A_frag[(k_frag + 1) % 2][6],
                           A_frag[(k_frag + 1) % 2][7],
                           A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
                    lds128(B_frag[(k_frag + 1) % 2][0],
                           B_frag[(k_frag + 1) % 2][1],
                           B_frag[(k_frag + 1) % 2][2],
                           B_frag[(k_frag + 1) % 2][3],
                           B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
                    lds128(B_frag[(k_frag + 1) % 2][4],
                           B_frag[(k_frag + 1) % 2][5],
                           B_frag[(k_frag + 1) % 2][6],
                           B_frag[(k_frag + 1) % 2][7],
                           B_lds_addr + ((k_frag + 1) % 8 * 128 + 32) * sizeof(float));

                    // load next A&B tile
                    if (k_frag == 0) {
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            ldg32_nc(A_ldg_reg[i],
                                     A_ldg_ptr + i * A_ldg_step,
                                     (A_ldg_guard & (1u << i)) != 0);
                        }

#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            ldg32_nc(B_ldg_reg[i],
                                     B_ldg_ptr + i * 32 * sizeof(float),
                                     (B_ldg_guard & (1u << i)) != 0);
                        }
                    }

                    // FFMA loop
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            if (conn[i][j]) {
                                C_frag[i][j] += A_frag[k_frag % 2][i] * B_frag[k_frag % 2][j];
                            }
                        }
                    }
                }
            }

            // FFMA for the last tile
#pragma unroll
            for (int k_frag = 0; k_frag < 8; ++k_frag) {
                if (k_frag < 7) {
                    // load next A&B fragment from shared memory to register
                    lds128(A_frag[(k_frag + 1) % 2][0],
                           A_frag[(k_frag + 1) % 2][1],
                           A_frag[(k_frag + 1) % 2][2],
                           A_frag[(k_frag + 1) % 2][3],
                           A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
                    lds128(A_frag[(k_frag + 1) % 2][4],
                           A_frag[(k_frag + 1) % 2][5],
                           A_frag[(k_frag + 1) % 2][6],
                           A_frag[(k_frag + 1) % 2][7],
                           A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
                    lds128(B_frag[(k_frag + 1) % 2][0],
                           B_frag[(k_frag + 1) % 2][1],
                           B_frag[(k_frag + 1) % 2][2],
                           B_frag[(k_frag + 1) % 2][3],
                           B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
                    lds128(B_frag[(k_frag + 1) % 2][4],
                           B_frag[(k_frag + 1) % 2][5],
                           B_frag[(k_frag + 1) % 2][6],
                           B_frag[(k_frag + 1) % 2][7],
                           B_lds_addr + ((k_frag + 1) % 8 * 128 + 32) * sizeof(float));
                }

                // FFMA loop
#pragma unroll
                for (int i = 0; i < 8; ++i) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        if (conn[i][j]) {
                            C_frag[i][j] += A_frag[k_frag % 2][i] * B_frag[k_frag % 2][j];
                        }
                    }
                }
            }

            // O = V @ C
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    O_frag[j] += V_frag[i] * C_frag[i][j];
                }
            }

            // O write back, reuse A&B tile shared memory buffer
            C_frag[0][0] = 0;
            if (threadIdx.x % 2 == 0) {
                uint32_t O_s_addr = smem_u32addr((float *) (smem + threadIdx.x / 2 * 4));
                sts32(C_frag[0][0], O_s_addr);
            }
            __syncthreads();
            float *O_smem = (float *) (smem) + (warp_id % 2) * 64 + mma_tid_x * 4;
#pragma unroll
            for (int i = 0; i < 2; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    atomicAdd(O_smem + 32 * i + j, O_frag[i * 4 + j]);
                }
            }
            __syncthreads();

            const float *O_lds_ptr = (float *) (smem) + threadIdx.x / 2;
            if (threadIdx.x % 2 == 0) {
                uint32_t n_idx = blockIdx.x * 128 + threadIdx.x / 2;
                if (n_idx < n) {
                    atomicAdd(O + n_idx, O_lds_ptr[0]);
                }
            }
        }


    }  // namespace


    void vector_matmul_mask(cudaStream_t stream,
                            void **buffers,
                            const char *opaque,
                            std::size_t opaque_len) {
        // size
        const MatMulDescriptor &d = *UnpackDescriptor<MatMulDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const std::uint32_t seed = d.seed;
        const float p = d.p;

        // input and output data
        const float *V = reinterpret_cast<const float *>(buffers[0]);
        const float *L = reinterpret_cast<const float *>(buffers[1]);
        const float *R = reinterpret_cast<const float *>(buffers[2]);
        float *O = reinterpret_cast<float *>(buffers[3]);

        // call kernel
        dim3 grid((n + 127) / 128, (m + 127) / 128);
        cudaMemset(O, 0, sizeof(float) * n);
        vector_matmul_mask_kernel<<<grid, 256, 0, stream>>>(V, L, R, O,
                                                            m, n, k,
                                                            k * sizeof(float),
                                                            n * sizeof(float) * 8,
                                                            seed,
                                                            p);
        ThrowIfError(cudaGetLastError());
    }



}  // namespace brainpylib
