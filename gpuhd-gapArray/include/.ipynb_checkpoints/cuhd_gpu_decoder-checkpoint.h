/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_GPU_DECODER_
#define CUHD_GPU_DECODER_

#include "cuhd_constants.h"
#include "cuhd_gpu_input_buffer.h"
#include "cuhd_gpu_output_buffer.h"
#include "cuhd_gpu_codetable.h"
#include "cuhd_gpu_decoder_memory.h"
#include "cuhd_subsequence_sync_point.h"

namespace cuhd {
    class CUHDGPUDecoder {
        public:
            // 9-argument signature (Original/Standard Huffman)
            static void decode(std::shared_ptr<cuhd::CUHDGPUInputBuffer> input,
                size_t input_size,
                std::shared_ptr<cuhd::CUHDGPUOutputBuffer> output,
                size_t output_size,
                std::shared_ptr<cuhd::CUHDGPUCodetable> table,
                std::shared_ptr<cuhd::CUHDGPUDecoderMemory> aux,
                size_t max_codeword_length,
                size_t preferred_subsequence_size,
                size_t threads_per_block);

            // 12-argument signature (Gap Array Huffman) - Added for gpuhd-gapArray
            static void decode(std::shared_ptr<cuhd::CUHDGPUInputBuffer> input,
                size_t input_size, // Total size of buffer (Gap Array + Data)
                std::shared_ptr<cuhd::CUHDGPUOutputBuffer> output,
                size_t output_size,
                std::shared_ptr<cuhd::CUHDGPUCodetable> table,
                std::shared_ptr<cuhd::CUHDGPUDecoderMemory> aux,
                size_t max_codeword_length,
                size_t preferred_subsequence_size,
                size_t threads_per_block,
                size_t compressed_data_size_units, // **UPDATED: New 10th argument (Data size in units)**
                size_t gap_element_num,            // New: 11th argument
                size_t gap_array_size_units);      // New: 12th argument
    };
}

#endif /* CUHD_GPU_DECODER_ */