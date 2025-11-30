/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cuda_helper.cuh"
#include "gpu_physical_hash_join.hpp"
#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"

namespace duckdb {

template <int B, int I>
__global__ void scan_left_unmatched(uint8_t* matched_flags, unsigned long long* count, uint64_t N, 
                uint64_t *row_ids, int is_count) {

    typedef cub::BlockScan<int, B> BlockScanInt;

    __shared__ union TempStorage
    {
        typename BlockScanInt::TempStorage scan;
    } temp_storage;

    int selection_flags[I];
    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    int t_count = 0;
    int c_t_count = 0;
    __shared__ uint64_t block_off;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        selection_flags[ITEM] = 0;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        if (threadIdx.x + (ITEM * B) < num_tile_items) {
            uint64_t idx = tile_offset + threadIdx.x + (ITEM * B);
            if (matched_flags[idx] == 0) {
                selection_flags[ITEM] = 1;
                t_count++;
            }
        }
    }

    __syncthreads();

    BlockScanInt(temp_storage.scan).ExclusiveSum(t_count, c_t_count);
    if(threadIdx.x == blockDim.x - 1) {
        block_off = atomicAdd(count, (unsigned long long) t_count+c_t_count);
    }

    __syncthreads();

    if (is_count) return;

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            if(selection_flags[ITEM]) {
                uint64_t offset = block_off + c_t_count++;
                row_ids[offset] = tile_offset + threadIdx.x + ITEM * B;
            }
        }
    }
}

template
__global__ void scan_left_unmatched<BLOCK_THREADS, ITEMS_PER_THREAD>(uint8_t* matched_flags, unsigned long long* count, uint64_t N, 
                uint64_t *row_ids, int is_count);

void scanUnmatchedLHSRows(uint8_t* matched_flags, uint64_t N, uint64_t* &row_ids, uint64_t* &count) {
    CHECK_ERROR();
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    if (N == 0) {
        uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
        h_count[0] = 0;
        count = h_count;
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }
    SIRIUS_LOG_DEBUG("Launching Scan Left Unmatched Kernel");
    SETUP_TIMING();
    START_TIMER();
    count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
    cudaMemset(count, 0, sizeof(uint64_t));

    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    CHECK_ERROR();
    scan_left_unmatched<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(matched_flags, (unsigned long long*) count, N, nullptr, 1);
    CHECK_ERROR();
    cudaDeviceSynchronize();

    uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    if (h_count[0] == 0) {
        SIRIUS_LOG_DEBUG("No unmatched LHS rows");
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(count), 0);
        count = h_count;
        STOP_TIMER();
        return;
    }
    row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
    cudaMemset(count, 0, sizeof(uint64_t));
    scan_left_unmatched<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(matched_flags, (unsigned long long*) count, N, row_ids, 0);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    SIRIUS_LOG_DEBUG("Scan Left Unmatched Count: {}", h_count[0]);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(count), 0);
    count = h_count;
    STOP_TIMER();
}

} // namespace duckdb

