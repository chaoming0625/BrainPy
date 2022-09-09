// This file contains the GPU implementation of our op. It's a pretty typical CUDA kernel
// and I make no promises about the quality of the code or the choices made therein, but
// it should get the point across.

#include "event_mmm_gpu.h"

namespace brainpy_lib {

    namespace {


        __device__ __inline__
        bool check_load_sm(bool sp_load, bool *syn_arrived, const int num) {
            if (sp_load) {
#pragma unroll
                for (int i = 0; i < num; ++i) {
                    if (syn_arrived[i]) {
                        return true;
                    }
                }
            }
            return false;
        }


        // The operator processing "a real-value matrix" multiply "a mask matrix":
        //    R @ Mask
        //
        // Assumptions:
        // 1. K_TILE * M_TILE <= 2K
        // 2. K_TILE * M_TILE == BLOCK_DIM * 4
        // 3. N_TILE >= M_TILE * 2
        template<const int K_TILE,
                const int M_TILE,
                const int N_THREAD,
                const int SM_K_STRIDE,
                const int BLOCK_DIM>
        __global__ void event_mmm_8K_sm_kernel(
                const std::uint32_t seed,
                const bool *V,
                const float *A,
                const uint32_t m,
                const uint32_t n,
                const uint32_t k,
                const float log_p, // probability: log(1-p)
                float *O
        ) {
            // shared memory
            __shared__ __align__
            (16 * 1024)
            char smem[16 * 1024];
            float *A_smem = reinterpret_cast<float *>(smem);
            __shared__ bool SP_smem[2][BLOCK_DIM];  // blockDim.x >= M_TILE * 2

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

            // start position of K axis
            const uint32_t k_id_start = blockIdx.y * K_TILE;

            // position of n
            const uint32_t n_tid = threadIdx.x + blockIdx.x * blockDim.x;
            const uint32_t n_id = n_tid * N_THREAD;

            // position of K axis to read A
            const uint32_t y_id = threadIdx.x / M_TILE * 4 + k_id_start;

            // V load register
            bool SP_ldg_reg;
            // A load register
            float A_ldg_reg[4];
            // A_tile ldg pointer
            const char *A_ldg_ptr = (const char *) (A + y_id * m + threadIdx.x % M_TILE);

            // number of m tile to read A
            uint32_t num_m_tile_loop = (m + M_TILE - 1) / M_TILE - 1;

            // random state
            curandState state;
            curand_init(seed + n_tid, 0, 0, &state);

            // synapse state
            int syn_arrival_id[N_THREAD];
            bool syn_arrived[2][N_THREAD]; // whether spike arrives at the current or next step (m axis)
#pragma unroll
            for (int i = 0; i < N_THREAD; ++i) {
                syn_arrival_id[i] = (int) ceil(log(curand_uniform(&state)) / log_p);
                syn_arrived[0][i] = (syn_arrival_id[i] == 1);
                if (syn_arrived[0][i]) {
                    syn_arrival_id[i] += (int) ceil(log(curand_uniform(&state)) / log_p);
                }
                syn_arrived[1][i] = (syn_arrival_id[i] == 2);
            }

            // ldg_guard to avoid LDG out of bound
            uint32_t A_k_guard = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if ((y_id + i) < m) { A_k_guard |= (1u << i); }
            }
            // A_tile sts/lds pointer
            uint32_t A_sts_addr = smem_u32addr(A_smem +
                                               (threadIdx.x % M_TILE) * SM_K_STRIDE +
                                               threadIdx.x / M_TILE * 4);
            uint32_t A_lds_addr = smem_u32addr(A_smem);


            /*
             * Start working.
             */
            // load first tile of V
            uint32_t m_id_for_comp = 0;

            // load first 2*M_TILE spikes into shared memory
            if (threadIdx.x < m) {
                SP_ldg_reg = V[threadIdx.x];
            } else {
                SP_ldg_reg = false;
            }
            SP_smem[0][threadIdx.x] = SP_ldg_reg;
            __syncthreads();

            // load 1'st A_tile to sm according to the spike event
            bool next_A_ldg_load = SP_smem[0][threadIdx.x % M_TILE];
            if (next_A_ldg_load) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    bool guard = (A_k_guard & (1u << i)) != 0 && (threadIdx.x % M_TILE < m);
                    ldg32_nc_0(A_ldg_reg[i], A_ldg_ptr + i * m * sizeof(float), guard);
                }
                sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3], A_sts_addr);
            }
            __syncthreads();
            A_sts_addr ^= 0x2000;  // 0x十六进制，8K
            A_ldg_ptr += M_TILE * sizeof(float);  // 从左往右移动k
            int SP_lds_id = 0;
            int SP_sts_id = 1;

            // load 1'st fragment
            bool SP_at_cur_step = SP_smem[SP_lds_id][0];
            bool SP_at_next_step;
            if (check_load_sm(SP_at_cur_step, syn_arrived[0], N_THREAD)) {
#pragma unroll
                for (int i = 0; i < K_TILE; i += 4) {
                    lds128(A_frag[0][i], A_frag[0][i + 1], A_frag[0][i + 2], A_frag[0][i + 3],
                           A_lds_addr + i * sizeof(float));
                }
            }

            /*
             * num_m_tile_loop loop
             */

            for (; num_m_tile_loop > 0; --num_m_tile_loop) {
#pragma unroll
                for (int m_frag = 0; m_frag < M_TILE; ++m_frag) {

                    if (m_frag == M_TILE - 1) {
                        // store next A and V tile to shared memory
                        SP_smem[SP_sts_id][threadIdx.x] = SP_ldg_reg;
                        if (next_A_ldg_load) {
                            sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3], A_sts_addr);
                        }
                        __syncthreads();

                        // switch double buffer
                        A_lds_addr ^= 0x2000;
                        A_sts_addr ^= 0x2000;
                        SP_sts_id ^= 1;
                        SP_lds_id ^= 1;

                        // ldg pointer for next tile
                        A_ldg_ptr += M_TILE * sizeof(float);
                    }

                    SP_at_next_step = SP_smem[SP_lds_id][(m_frag + 1) % M_TILE];

                    // load next A fragment from shared memory
                    if (check_load_sm(SP_at_next_step, syn_arrived[(m_frag + 1) % 2], N_THREAD)) {
#pragma unroll
                        for (int i = 0; i < K_TILE; i += 4) {
                            lds128(A_frag[(m_frag + 1) % 2][i],
                                   A_frag[(m_frag + 1) % 2][i + 1],
                                   A_frag[(m_frag + 1) % 2][i + 2],
                                   A_frag[(m_frag + 1) % 2][i + 3],
                                   A_lds_addr + ((m_frag + 1) % M_TILE * SM_K_STRIDE + i) * sizeof(float));
                        }
                    }


                    if (m_frag == 0) {
                        // load next spike tile
                        if ((m_id_for_comp + M_TILE + threadIdx.x) < m) {
                            SP_ldg_reg = V[m_id_for_comp + M_TILE + threadIdx.x];
                        } else {
                            SP_ldg_reg = false;
                        }
                        next_A_ldg_load = SP_smem[SP_lds_id][M_TILE + threadIdx.x % M_TILE];
                        // load next A tile
                        if (next_A_ldg_load) {
                            bool A_m_guard = (m_id_for_comp + M_TILE + threadIdx.x % M_TILE) < m;
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                bool guard = ((A_k_guard & (1u << i)) != 0) && A_m_guard;
                                ldg32_nc(A_ldg_reg[i], A_ldg_ptr + i * m * sizeof(float), guard);
                            }
                        }
                    }

                    // if spike and synapse id arrived, FFMA
                    if (SP_at_cur_step) {
#pragma unroll
                        for (int i = 0; i < N_THREAD; ++i) {
                            if (syn_arrived[m_frag % 2][i]) {
#pragma unroll
                                for (int j = 0; j < K_TILE; ++j) {
                                    O_frag[j][i] += A_frag[m_frag % 2][j];
                                }
                            }
                        }
                    }

                    // get the next spike load indicator
                    SP_at_cur_step = SP_at_next_step;

                    // check synapse states at the next step,
                    // overwrite synapse states at the current step
#pragma unroll
                    for (int i = 0; i < N_THREAD; ++i) {
                        if (syn_arrived[(m_frag + 1) % 2][i]) {
                            syn_arrival_id[i] += (int) ceil(log(curand_uniform(&state)) / log_p);  // error
                        }
                        syn_arrived[m_frag % 2][i] = (syn_arrival_id[i] == m_id_for_comp + m_frag + 3);
                    }

                }
                m_id_for_comp += M_TILE;
            }

            // FFMA for the last tile
            for (int m_frag = 0; m_frag < (m % M_TILE); ++m_frag) {
                SP_at_next_step = SP_smem[SP_lds_id][m_frag + 1];
                if (m_frag < (m % M_TILE - 1)) {
                    // load next A fragment from shared memory
                    if (check_load_sm(SP_at_next_step, syn_arrived[(m_frag + 1) % 2], N_THREAD)) {
#pragma unroll
                        for (int i = 0; i < K_TILE; i += 4) {
                            lds128(A_frag[(m_frag + 1) % 2][i],
                                   A_frag[(m_frag + 1) % 2][i + 1],
                                   A_frag[(m_frag + 1) % 2][i + 2],
                                   A_frag[(m_frag + 1) % 2][i + 3],
                                   A_lds_addr + ((m_frag + 1) % M_TILE * SM_K_STRIDE + i) * sizeof(float));
                        }
                    }
                }

                // FFMA loop
                if (SP_at_cur_step) {
#pragma unroll
                    for (int i = 0; i < N_THREAD; ++i) {
                        if (syn_arrived[m_frag % 2][i]) {
#pragma unroll
                            for (int j = 0; j < K_TILE; ++j) {
                                O_frag[j][i] += A_frag[m_frag % 2][j];
                            }
                        }
                    }
                }

                // get the next spike load indicator
                SP_at_cur_step = SP_at_next_step;


                // check synapse states at the next step,
                // overwrite synapse states at the current step
#pragma unroll
                for (int i = 0; i < N_THREAD; ++i) {
                    if (syn_arrived[(m_frag + 1) % 2][i]) {
                        syn_arrival_id[i] += (int) ceil(log(curand_uniform(&state)) / log_p);
                    }
                    syn_arrived[m_frag % 2][i] = (syn_arrival_id[i] == m_id_for_comp + m_frag + 3);
                }
            }

            // O_tile write back
#pragma unroll
            for (int i = 0; i < K_TILE; ++i) {
#pragma unroll
                for (int j = 0; j < N_THREAD; ++j) {
                    if ((k_id_start + i < k) && (n_id + j < n)) {
                        O[(k_id_start + i) * n + n_id + j] = O_frag[i][j];
                    }
                }
            }
        }

        template<const int K_TILE, const int M_TILE, const int SM_K_STRIDE, const int BLOCK_DIM>
        __global__ void event_mmm_8Ksm_n1_kernel(
                const std::uint32_t seed,
                const bool *V,
                const float *A,
                const uint32_t m,
                const uint32_t n,
                const uint32_t k,
                const float log_p, // probability: log(1-p)
                float *O
        ) {
            // shared memory
            __shared__ __align__
            (16 * 1024)
            char smem[16 * 1024];
            float *A_smem = reinterpret_cast<float *>(smem);
            __shared__ bool SP_smem[2][BLOCK_DIM];  // blockDim.x >= M_TILE * 2

            // A register fragment
            float A_frag[2][K_TILE];

            // O register fragment
            float O_frag[K_TILE];
#pragma unroll
            for (int i = 0; i < K_TILE; ++i) {
                O_frag[i] = 0;
            }

            // start position of K axis
            const uint32_t k_id_start = blockIdx.y * K_TILE;

            // position of n
            const uint32_t n_id = threadIdx.x + blockIdx.x * blockDim.x;

            // position of K axis to read A
            const uint32_t y_id = threadIdx.x / M_TILE * 4 + k_id_start;

            // V load register
            bool SP_ldg_reg;
            // A load register
            float A_ldg_reg[4];
            // A_tile ldg pointer
            const char *A_ldg_ptr = (const char *) (A + y_id * m + threadIdx.x % M_TILE);

            // number of m tile to read A
            uint32_t num_m_tile_loop = (m + M_TILE - 1) / M_TILE - 1;

            // random state
            curandState state;
            curand_init(seed + n_id, 0, 0, &state);

            // synapse state
            int syn_arrival_id;
            bool syn_arrived[2]; // whether spike arrives at the current or next step (m axis)
            syn_arrival_id = (int) ceil(log(curand_uniform(&state)) / log_p);
            syn_arrived[0] = (syn_arrival_id == 1);
            if (syn_arrived[0]) {
                syn_arrival_id += (int) ceil(log(curand_uniform(&state)) / log_p);
            }
            syn_arrived[1] = (syn_arrival_id == 2);

            // ldg_guard to avoid LDG out of bound
            uint32_t A_k_guard = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if ((y_id + i) < m) { A_k_guard |= (1u << i); }
            }
            // A_tile sts/lds pointer
            uint32_t A_sts_addr = smem_u32addr(A_smem +
                                               (threadIdx.x % M_TILE) * SM_K_STRIDE +
                                               threadIdx.x / M_TILE * 4);
            uint32_t A_lds_addr = smem_u32addr(A_smem);


            /*
             * Start working.
             */
            // load first tile of V
            uint32_t m_id_for_comp = 0;

            // load first 2*M_TILE spikes into shared memory
            if (threadIdx.x < m) {
                SP_ldg_reg = V[threadIdx.x];
            } else {
                SP_ldg_reg = false;
            }
            SP_smem[0][threadIdx.x] = SP_ldg_reg;
            __syncthreads();

            // load 1'st A_tile to sm according to the spike event
            bool next_A_ldg_load = SP_smem[0][threadIdx.x % M_TILE];
            if (next_A_ldg_load) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    bool guard = (A_k_guard & (1u << i)) != 0 && (threadIdx.x % M_TILE < m);
                    ldg32_nc_0(A_ldg_reg[i], A_ldg_ptr + i * m * sizeof(float), guard);
                }
                sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3], A_sts_addr);
            }
            __syncthreads();
            A_sts_addr ^= 0x2000;  // 0x十六进制，8K
            A_ldg_ptr += M_TILE * sizeof(float);  // 从左往右移动k
            int SP_lds_id = 0;
            int SP_sts_id = 1;

            // load 1'st fragment
            bool SP_at_cur_step = SP_smem[SP_lds_id][0];
            bool SP_at_next_step;
            if ((SP_at_cur_step && syn_arrived[0])) {
#pragma unroll
                for (int i = 0; i < K_TILE; i += 4) {
                    lds128(A_frag[0][i], A_frag[0][i + 1], A_frag[0][i + 2], A_frag[0][i + 3],
                           A_lds_addr + i * sizeof(float));
                }
            }

            /*
             * num_m_tile_loop loop
             */

            for (; num_m_tile_loop > 0; --num_m_tile_loop) {
#pragma unroll
                for (int m_frag = 0; m_frag < M_TILE; ++m_frag) {

                    if (m_frag == M_TILE - 1) {
                        // store next A and V tile to shared memory
                        SP_smem[SP_sts_id][threadIdx.x] = SP_ldg_reg;
                        if (next_A_ldg_load) {
                            sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3], A_sts_addr);
                        }
                        __syncthreads();

                        // switch double buffer
                        A_lds_addr ^= 0x2000;
                        A_sts_addr ^= 0x2000;
                        SP_sts_id ^= 1;
                        SP_lds_id ^= 1;

                        // ldg pointer for next tile
                        A_ldg_ptr += M_TILE * sizeof(float);
                    }

                    SP_at_next_step = SP_smem[SP_lds_id][(m_frag + 1) % M_TILE];

                    // load next A fragment from shared memory
                    if ((SP_at_next_step && syn_arrived[(m_frag + 1) % 2])) {
#pragma unroll
                        for (int i = 0; i < K_TILE; i += 4) {
                            lds128(A_frag[(m_frag + 1) % 2][i],
                                   A_frag[(m_frag + 1) % 2][i + 1],
                                   A_frag[(m_frag + 1) % 2][i + 2],
                                   A_frag[(m_frag + 1) % 2][i + 3],
                                   A_lds_addr + ((m_frag + 1) % M_TILE * SM_K_STRIDE + i) * sizeof(float));
                        }
                    }


                    if (m_frag == 0) {
                        // load next spike tile
                        if ((m_id_for_comp + M_TILE + threadIdx.x) < m) {
                            SP_ldg_reg = V[m_id_for_comp + M_TILE + threadIdx.x];
                        } else {
                            SP_ldg_reg = false;
                        }
                        next_A_ldg_load = SP_smem[SP_lds_id][M_TILE + threadIdx.x % M_TILE];
                        // load next A tile
                        if (next_A_ldg_load) {
                            bool A_m_guard = (m_id_for_comp + M_TILE + threadIdx.x % M_TILE) < m;
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                bool guard = ((A_k_guard & (1u << i)) != 0) && A_m_guard;
                                ldg32_nc(A_ldg_reg[i], A_ldg_ptr + i * m * sizeof(float), guard);
                            }
                        }
                    }

                    // if spike and synapse id arrived, FFMA
                    if (SP_at_cur_step && syn_arrived[m_frag % 2]) {
#pragma unroll
                        for (int j = 0; j < K_TILE; ++j) {
                            O_frag[j] += A_frag[m_frag % 2][j];
                        }
                    }

                    // get the next spike load indicator
                    SP_at_cur_step = SP_at_next_step;

                    // check synapse states at the next step,
                    // overwrite synapse states at the current step
                    if (syn_arrived[(m_frag + 1) % 2]) {
                        syn_arrival_id += (int) ceil(log(curand_uniform(&state)) / log_p);
                    }
                    syn_arrived[m_frag % 2] = (syn_arrival_id == m_id_for_comp + m_frag + 3);

                }
                m_id_for_comp += M_TILE;
            }

            // FFMA for the last tile
            for (int m_frag = 0; m_frag < (m % M_TILE); ++m_frag) {
                SP_at_next_step = SP_smem[SP_lds_id][m_frag + 1];
                if (m_frag < (m % M_TILE - 1)) {
                    // load next A fragment from shared memory
                    if (SP_at_next_step && syn_arrived[(m_frag + 1) % 2]) {
#pragma unroll
                        for (int i = 0; i < K_TILE; i += 4) {
                            lds128(A_frag[(m_frag + 1) % 2][i],
                                   A_frag[(m_frag + 1) % 2][i + 1],
                                   A_frag[(m_frag + 1) % 2][i + 2],
                                   A_frag[(m_frag + 1) % 2][i + 3],
                                   A_lds_addr + ((m_frag + 1) % M_TILE * SM_K_STRIDE + i) * sizeof(float));
                        }
                    }
                }

                // FFMA loop
                if (SP_at_cur_step && syn_arrived[m_frag % 2]) {
#pragma unroll
                    for (int j = 0; j < K_TILE; ++j) {
                        O_frag[j] += A_frag[m_frag % 2][j];
                    }
                }

                // get the next spike load indicator
                SP_at_cur_step = SP_at_next_step;

                // check synapse states at the next step,
                // overwrite synapse states at the current step
                if (syn_arrived[(m_frag + 1) % 2]) {
                    syn_arrival_id += (int) ceil(log(curand_uniform(&state)) / log_p);
                }
                syn_arrived[m_frag % 2] = (syn_arrival_id == m_id_for_comp + m_frag + 3);
            }

            // O_tile write back
            if (n_id < n) {
#pragma unroll
                for (int i = 0; i < K_TILE; ++i) {
                    if (k_id_start + i < k) {
                        O[(k_id_start + i) * n + n_id] = O_frag[i];
                    }
                }
            }
        }


    }  // namespace



//    void event_mmm_8K_1x8x128x256(
//            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
//    ) {
//        // size
//        const MatMulDescriptor &d = *UnpackDescriptor<MatMulDescriptor>(opaque, opaque_len);
//        const std::uint32_t m = d.m;
//        const std::uint32_t k = d.k;
//        const std::uint32_t n = d.n;
//        const std::uint32_t seed = d.seed;
//        const float log_p = d.p;
//
//        // input and output data
////        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[0]);
////        const bool *V = reinterpret_cast<const bool *>(buffers[1]);
////        const float *A = reinterpret_cast<const float *>(buffers[2]);
////        float *O = reinterpret_cast<float *>(buffers[3]);
//
//        const bool *V = reinterpret_cast<const bool *>(buffers[0]);
//        const float *A = reinterpret_cast<const float *>(buffers[1]);
//        float *O = reinterpret_cast<float *>(buffers[2]);
//
//        // call kernel
//        cudaMemset(O, 0, sizeof(float) * n * k);
//        dim3 grid((n + 255) / 256, (k + 7) / 8);
//        event_mmm_8K_sm_kernel<8, 128, 1, 12, 256><<<grid, 256, 0, stream>>>(seed, V, A, m, n, k, log_p, O);
//        ThrowIfError(cudaGetLastError());
//    }

    void event_mmm_8K_1x8x128x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        // size
        const MatMulDescriptor &d = *UnpackDescriptor<MatMulDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const std::uint32_t seed = d.seed;
        const float log_p = d.p;

        const bool *V = reinterpret_cast<const bool *>(buffers[0]);
        const float *A = reinterpret_cast<const float *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 255) / 256, (k + 7) / 8);
        event_mmm_8Ksm_n1_kernel<8, 128, 12, 256><<<grid, 256, 0, stream>>>(seed, V, A, m, n, k, log_p, O);
        ThrowIfError(cudaGetLastError());
    }


    void event_mmm_8K_4x8x128x256(
            cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
    ) {
        // size
        const MatMulDescriptor &d = *UnpackDescriptor<MatMulDescriptor>(opaque, opaque_len);
        const std::uint32_t m = d.m;
        const std::uint32_t k = d.k;
        const std::uint32_t n = d.n;
        const std::uint32_t seed = d.seed;
        const float log_p = d.p;

        // input and output data
//        const curandState *rng_states = reinterpret_cast<const curandState *>(buffers[0]);
//        const bool *V = reinterpret_cast<const bool *>(buffers[1]);
//        const float *A = reinterpret_cast<const float *>(buffers[2]);
//        float *O = reinterpret_cast<float *>(buffers[3]);

        const bool *V = reinterpret_cast<const bool *>(buffers[0]);
        const float *A = reinterpret_cast<const float *>(buffers[1]);
        float *O = reinterpret_cast<float *>(buffers[2]);

        // call kernel
        cudaMemset(O, 0, sizeof(float) * n * k);
        dim3 grid((n + 1023) / 1024, (k + 7) / 8);
        event_mmm_8K_sm_kernel<8, 128, 4, 12, 256><<<grid, 256, 0, stream>>>(seed, V, A, m, n, k, log_p, O);
        ThrowIfError(cudaGetLastError());
    }


}  // namespace brainpylib
