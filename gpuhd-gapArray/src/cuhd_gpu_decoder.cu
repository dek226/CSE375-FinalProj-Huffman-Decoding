/*****************************************************************************
 *
 * CUHD - A massively parallel Huffman decoder
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#include "cuhd_gpu_decoder.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

__device__ __forceinline__ void decode_subsequence(
    std::uint32_t subsequence_size,
    std::uint32_t current_subsequence,
    UNIT_TYPE mask,
    std::uint32_t shift,
    std::uint32_t start_bit,
    std::uint32_t &in_pos,
    const UNIT_TYPE* __restrict__ in_ptr,
    UNIT_TYPE &window,
    UNIT_TYPE &next,
    std::uint32_t &last_word_unit,
    std::uint32_t &last_word_bit,
    std::uint32_t &num_symbols,
    std::uint32_t &out_pos,
    SYMBOL_TYPE* out_ptr,
    std::uint32_t &next_out_pos,
    const cuhd::CUHDCodetableItemSingle* __restrict__ table,
    const std::uint32_t bits_in_unit,
    std::uint32_t &last_at,
    bool overflow,
    bool write_output,
    const std::uint32_t compressed_data_size_units) { // **NEW ARGUMENT: for boundary checks**

    // Helper function to safely read the input unit
    auto read_unit = [&] (std::uint32_t index) -> UNIT_TYPE {
        return (index < compressed_data_size_units) ? in_ptr[index] : 0;
    };
    
    // current unit in this subsequence
    std::uint32_t current_unit = 0;
    
    // current bit position in unit
    std::uint32_t at = start_bit;

    // number of symbols found in this subsequence
    std::uint32_t num_symbols_l = 0;

    // --- OVERFLOW BLOCK FIX (Using safe read_unit) ---
    // perform overflow from previous subsequence
    if(overflow && current_subsequence > 0) {

        // shift to start
        UNIT_TYPE copy_next = next;
        copy_next >>= bits_in_unit - at;

        next <<= at;
        window <<= at;
        window += copy_next;

        // decode first symbol
        std::uint32_t taken = table[(window & mask) >> shift].num_bits;

        copy_next = next;
        copy_next >>= bits_in_unit - taken;

        next <<= taken;
        window <<= taken;
        at += taken;
        window += copy_next;

        // overflow
        if(at > bits_in_unit) {
            ++in_pos;
            window = read_unit(in_pos);
            next = read_unit(in_pos + 1);
            at -= bits_in_unit;
            window <<= at;
            next <<= at;

            copy_next = read_unit(in_pos + 1);
            copy_next >>= bits_in_unit - at;
            window += copy_next;
        }

        else {
            ++in_pos;
            window = read_unit(in_pos);
            next = read_unit(in_pos + 1);
            at = 0;
        }
    }
    // --- END OVERFLOW BLOCK FIX ---
    
    while(current_unit < subsequence_size) {
        
        while(at < bits_in_unit) {
            const cuhd::CUHDCodetableItemSingle hit =
                table[(window & mask) >> shift];
            
            // decode a symbol
            std::uint32_t taken = hit.num_bits;
            ++num_symbols_l;

            if(write_output) {
                if(out_pos < next_out_pos) {
                    out_ptr[out_pos] = hit.symbol;
                    ++out_pos;
                }
            }
            
            UNIT_TYPE copy_next = next;
            copy_next >>= bits_in_unit - taken;

            next <<= taken;
            window <<= taken;
            last_word_bit = at;
            at += taken;
            window += copy_next;
            last_word_unit = current_unit;
        }
        
        // refill decoder window if necessary
        ++current_unit;
        ++in_pos;
        
        // ** CRITICAL FIX: Use safe read_unit to prevent illegal memory access **
        window = read_unit(in_pos);
        next = read_unit(in_pos + 1);
        // ** END CRITICAL FIX **
        
        if(at == bits_in_unit) {
            at = 0;
        }

        else {
            at -= bits_in_unit;
            window <<= at;
            next <<= at;
            
            UNIT_TYPE copy_next = read_unit(in_pos + 1); // Use safe read here too!
            copy_next >>= bits_in_unit - at;
            window += copy_next;
        }
    }
    
    num_symbols = num_symbols_l;
    last_at = at;
}

__global__ void phase3_copy_num_symbols_from_sync_points_to_aux(
    std::uint32_t total_num_subsequences,
    const uint4* __restrict__ sync_points,
    std::uint32_t* subsequence_output_sizes) {

    const std::uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid < total_num_subsequences) {
        subsequence_output_sizes[gid] = sync_points[gid].z;
    }
}

__global__ void phase3_copy_num_symbols_from_aux_to_sync_points(
    std::uint32_t total_num_subsequences,
    uint4* sync_points,
    const std::uint32_t* __restrict__ subsequence_output_sizes) {

    const std::uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid < total_num_subsequences) {
        sync_points[gid].z = subsequence_output_sizes[gid];
    }
}

// NOTE: The signature is updated to include compressed_data_size_units
__global__ void phase4_decode_write_output_gap_array(
    std::uint32_t subsequence_size,
    std::uint32_t total_num_subsequences,
    std::uint32_t table_size,
    const UNIT_TYPE* __restrict__ in_ptr, // Compressed data pointer (adjusted by gap_array_size_units)
    SYMBOL_TYPE* out_ptr,
    std::uint32_t output_size,
    cuhd::CUHDCodetableItemSingle* table,
    const uint4* __restrict__ sync_points, // Contains output offsets (sync_points[gid].z)
    const std::uint32_t bits_in_unit,
    const UNIT_TYPE* __restrict__ gap_array_ptr,
    const std::uint32_t compressed_data_size_units, // **NEW: 11th argument**
    const std::uint32_t gap_array_size_units) {
    
    const std::uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid < total_num_subsequences) {
        
        // mask, shift, last_word_unit, last_word_bit, num_symbols, last_at remain...
        const UNIT_TYPE mask = ~(((UNIT_TYPE) (0) - 1) >> table_size);
        const size_t shift = bits_in_unit - table_size;

        std::uint32_t last_word_unit = 0;
        std::uint32_t last_word_bit = 0;
        std::uint32_t num_symbols = 0;
        std::uint32_t last_at = 0;
        
        // This is the current Global ID (subsequence index)
        std::uint32_t current_subsequence = gid;
        
        // 1. Get the starting bit offset from the Gap Array
        size_t bit_offset_total = gap_array_ptr[gid];
        
        // 2. Calculate the starting UNIT index (relative to the start of the compressed data)
        std::uint32_t compressed_data_relative_unit = bit_offset_total / bits_in_unit;

        // 3. The starting input position (index relative to in_ptr_data)
        std::uint32_t in_pos = compressed_data_relative_unit;

        // 4. Calculate the starting bit position within that unit (start)
        std::uint32_t start = bit_offset_total % bits_in_unit;

        
        // <<< GET OUTPUT START AND END POSITION >>>
        uint4 sync_point = sync_points[current_subsequence];
        uint4 next_sync_point = sync_points[current_subsequence + 1];
        
        std::uint32_t out_pos = sync_point.z;
        std::uint32_t next_out_pos = gid == total_num_subsequences - 1 ?
            output_size : next_sync_point.z;
        
        // Helper function to safely read the input unit for initialization
        auto read_unit = [&] (std::uint32_t index) -> UNIT_TYPE {
            return (index < compressed_data_size_units) ? in_ptr[index] : 0;
        };
        
        // sliding window initialized using the correct in_pos (Now using safe read)
        UNIT_TYPE window = read_unit(in_pos);
        UNIT_TYPE next = read_unit(in_pos + 1);
        
        // IMPORTANT: Shift window and next unit to align the start bit to the MSB position.
        window <<= start;
        UNIT_TYPE copy_next = next;
        copy_next >>= bits_in_unit - start;
        window += copy_next;
        next <<= start;

        // Decode the subsequence. start_bit is 0, overflow is false.
        decode_subsequence(subsequence_size, current_subsequence, mask, shift,
            0, // start_bit is 0 since we pre-shifted
            in_pos, in_ptr, window, next,
            last_word_unit, last_word_bit, num_symbols, out_pos, out_ptr,
            next_out_pos, table, bits_in_unit, last_at, false, true,
            compressed_data_size_units); // **Pass new argument**
    }
}

void cuhd::CUHDGPUDecoder::decode(
    std::shared_ptr<cuhd::CUHDGPUInputBuffer> input,
    size_t input_size,
    std::shared_ptr<cuhd::CUHDGPUOutputBuffer> output,
    size_t output_size,
    std::shared_ptr<cuhd::CUHDGPUCodetable> table,
    std::shared_ptr<cuhd::CUHDGPUDecoderMemory> aux,
    size_t max_codeword_length,
    size_t preferred_subsequence_size,
    size_t threads_per_block,
    // --- HOST FUNCTION SIGNATURE FIX ---
    size_t compressed_data_size_units_host, // **New: 10th argument (Replaces gap_array_start_unit)**
    size_t gap_element_num,                 // New: 11th argument (Gap array element count)
    size_t gap_array_size_units)            // New: 12th argument (Gap array size in UNIT_TYPE)
    // ----------------------------------
{
    // The core difference: The Gap Array method skips Phase 1 (decode) and Phase 2 (synchronization)
    // and uses the Gap Array as a lookup table for the bit-stream start points.

    UNIT_TYPE* in_ptr_full = input->get();
    SYMBOL_TYPE* out_ptr = output->get();
    cuhd::CUHDCodetableItemSingle* table_ptr = table->get();

    uint4* sync_info = reinterpret_cast<uint4*>(aux->get_sync_info());
    std::uint32_t* output_sizes = aux->get_output_sizes();

    // The compressed data *starts* after the Gap Array.
    // The input buffer contains [Gap Array | Compressed Data].
    UNIT_TYPE* in_ptr_data = in_ptr_full + gap_array_size_units;
    
    // Use the argument passed from the host demo (which calculated TotalSize - GapArraySize)
    const size_t compressed_data_size_units = input_size - gap_array_size_units;

    // Use the Gap Array for the starting positions
    UNIT_TYPE* gap_array_ptr = in_ptr_full;
    
    // The number of subsequences is equal to the number of gap array elements.
    size_t num_subseq = gap_element_num;
    size_t num_sequences = SDIV(num_subseq, threads_per_block);

    const std::uint32_t bits_in_unit = sizeof(UNIT_TYPE) * 8;
    // --- REMOVE PHASE 1 AND PHASE 2 CALLS ---
    // (They are no longer needed as the Gap Array provides synchronization)

    // launch phase 3 (parallel prefix sum)
    thrust::device_ptr<std::uint32_t> thrust_sync_points(output_sizes);

    // This kernel ASSUMES sync_info (z component) contains the correct symbol counts from the encoder.
    phase3_copy_num_symbols_from_sync_points_to_aux<<<
        num_sequences, threads_per_block>>>(num_subseq, sync_info, output_sizes);
    CUERR

    thrust::exclusive_scan(thrust_sync_points,
        thrust_sync_points + num_subseq, thrust_sync_points);

    phase3_copy_num_symbols_from_aux_to_sync_points<<<
        num_sequences, threads_per_block>>>(num_subseq, sync_info, output_sizes);
    CUERR
    
    // launch phase 4 (final decoding using Gap Array)
    phase4_decode_write_output_gap_array<<<num_sequences, threads_per_block>>>(
                preferred_subsequence_size,
                num_subseq,
                max_codeword_length,
                in_ptr_data, // Pass the pointer to the START of the compressed DATA
                out_ptr,
                output_size,
                table_ptr,
                sync_info,
                bits_in_unit,
                gap_array_ptr,
                (std::uint32_t)compressed_data_size_units, // **Pass size of compressed data**
                (std::uint32_t)gap_array_size_units); // Pass Gap Array size (in units)
    CUERR
}