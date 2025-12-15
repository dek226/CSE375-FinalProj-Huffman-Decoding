#include <iostream>
#include <algorithm>
#include <random>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#include <memory>
#include <cmath> // For std::ceil
#include <chrono> // Required for TIMER_START/STOP macros

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

// --- REQUIRED EXTERNAL HEADERS ---
#include "llhuffman_encoder.h"
#include "llhuffman_encoder_table.h"

#include "cuhd_input_buffer.h"
#include "cuhd_output_buffer.h"
#include "cuhd_gpu_memory_buffer.h"
#include "cuhd_gpu_input_buffer.h"
#include "cuhd_gpu_output_buffer.h"
#include "cuhd_codetable.h"
#include "cuhd_gpu_codetable.h"
#include "cuhd_cuda_definitions.h" // Assuming this defines MAX_CODEWORD_LENGTH, etc.
#include "cuhd_gpu_decoder.h"
#include "cuhd_gpu_decoder_memory.h"
#include "cuhd_util.h"
// ---------------------------------

// --- ASSUMED CUHD CONSTANTS FOR GAP ARRAY CALCULATION ---
#define UNIT_SIZE_BITS 32           // Size of UNIT_TYPE in bits
#define GAP_LENGTH_MAX 4            // Max bits needed to store the bit offset (e.g., 0-15)
#define GAP_FAC_NUM (UNIT_SIZE_BITS / GAP_LENGTH_MAX) // How many gap offsets fit in one UNIT_TYPE (32/4 = 8)

// subsequence size
#define SUBSEQ_SIZE 4

// threads per block
#define NUM_THREADS 128

// performance parameter, high = high performance, low = low memory consumption
#define DEVICE_PREF 9

// Constant for the data file name
#define INPUT_FILENAME "data.bin"

// Helper function to calculate Gap Array size (since this logic is internal to the encoder)
void calculate_gap_array_metadata(
    const llhuff::LLHuffmanEncoderTable* enc_table,  
    int& out_gap_element_num,
    size_t& out_gap_array_size_units) {

    // --- GAP ARRAY CALCULATION ASSUMPTION ---
    // The reference repository uses 1KB (8192 bits) segment size.
    const size_t SEGMENT_SIZE_BITS = 8192; 
    
    // Convert compressed size from UNIT_TYPEs to bits
    size_t compressed_size_bits = enc_table->compressed_size * sizeof(UNIT_TYPE) * 8;

    // Number of elements is the number of segments in the compressed data
    out_gap_element_num = std::ceil((double)compressed_size_bits / SEGMENT_SIZE_BITS);
    
    // Calculate the size of the Gap Array in UNIT_TYPEs
    // Each gap element is GAP_LENGTH_MAX bits (4 bits)
    // GAP_FAC_NUM (8) gap elements are packed into one UNIT_TYPE (32 bits)
    out_gap_array_size_units = std::ceil((double)out_gap_element_num / GAP_FAC_NUM);
}

int main(int argc, char** argv) {
    // name of the binary file
    const char* bin = argv[0];
    
    // compute device to use
    const std::int64_t compute_device_id = atoi(argv[1]);
    
    // input size in MB (used as the default size if no file is present)
    const long int cli_size_bytes = atoi(argv[2]) * 1024 * 1024;
    long int size = cli_size_bytes; // Actual size, may be overwritten by file size
    
    if(argc != 3 || compute_device_id < 0 || size < 1) {
        std::cout << "USAGE: " << bin << " <compute device index> "
        << "<size of input in megabytes>" << std::endl;
        return 1;
    }

    // vector for storing time measurements
    std::vector<std::pair<std::string, size_t>> timings;
    
    // generate random data for testing
    std::vector<SYMBOL_TYPE> buffer;

    // ** START DATA LOADING/GENERATION BLOCK **
    
    std::ifstream infile(INPUT_FILENAME, std::ios::binary | std::ios::ate);

    if (infile.is_open()) {
        // File exists: Read data and update size
        size = infile.tellg();
        infile.seekg(0, std::ios::beg);
        buffer.resize(size);
        infile.read(reinterpret_cast<char*>(buffer.data()), size);
        infile.close();
        std::cout << "Input source: Read " << (size / (1024.0 * 1024.0)) << " MB from file: " << INPUT_FILENAME << std::endl;
    } else {
        // File missing: Use CLI size, generate random data (original logic), and save
        buffer.resize(size);

        std::independent_bits_engine<
            std::linear_congruential_engine<
            std::uint_fast32_t, 16807, 0, 2147483647>, 
            sizeof(SYMBOL_TYPE) * 8, SYMBOL_TYPE> gen;
        std::binomial_distribution<> dist(
            std::numeric_limits<SYMBOL_TYPE>::max(), 0.5);

        TIMER_START(timings, "generating random data")
            std::generate(begin(buffer), end(buffer), [&](){return dist(gen);});
        TIMER_STOP

        // Save to file
        std::ofstream outfile(INPUT_FILENAME, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(buffer.data()), size);
            outfile.close();
            std::cout << "Data saved to '" << INPUT_FILENAME << "'." << std::endl;
        } else {
            std::cerr << "Warning: Could not save generated data to '" << INPUT_FILENAME << "'." << std::endl;
        }
    }
    // ** END DATA LOADING/GENERATION BLOCK **

    std::shared_ptr<std::vector<llhuff::LLHuffmanEncoder::Symbol>> lengths;
    std::shared_ptr<llhuff::LLHuffmanEncoderTable> enc_table;
    std::shared_ptr<cuhd::CUHDCodetable> dec_table;
    
    // compress the random data using length-limited Huffman coding
    TIMER_START(timings, "generating coding tables")

        // determine optimal lengths for the codewords to be generated
        lengths = llhuff::LLHuffmanEncoder::get_symbol_lengths(
            buffer.data(), size);
    
        // generate encoder table (this function will calculate compressed_size)
        enc_table = llhuff::LLHuffmanEncoder::get_encoder_table(lengths);
    
        // generate decoder table
        dec_table = llhuff::LLHuffmanEncoder::get_decoder_table(enc_table);
    TIMER_STOP
    
    // --- GAP ARRAY IMPLEMENTATION CHANGES START HERE (HOST) ---

    // 1. Calculate Gap Array Metadata based on the encoder table's compressed size
    int gap_element_num = 0;
    size_t gap_array_size_units = 0;
    
    // Assuming the encoder calculated enc_table->compressed_size correctly.
    calculate_gap_array_metadata(enc_table.get(), gap_element_num, gap_array_size_units);


    // 2. Calculate TOTAL size needed for the buffer: Gap Array + Compressed Data
    const size_t total_compressed_units = 
        enc_table->compressed_size + gap_array_size_units;
    
    // The starting unit index of the compressed data stream within the total buffer.
    // This value is CRITICAL for the CUDA kernel to skip the Gap Array.
    const std::uint32_t compressed_data_start_unit = (std::uint32_t)gap_array_size_units;

    // --- DEBUG: Print Gap Array Metadata ---
    std::cout << "\n--- DEBUG: Gap Array Metadata ---" << std::endl;
    std::cout << "Compressed Size (Units): " << enc_table->compressed_size << std::endl;
    std::cout << "Gap Elements (# segments): " << gap_element_num << std::endl;
    std::cout << "Gap Array Size (Units): " << gap_array_size_units << std::endl;
    std::cout << "Total Buffer Size (Units): " << total_compressed_units << std::endl;
    std::cout << "Compressed Data Start Unit Index: " << compressed_data_start_unit << std::endl; // New debug
    std::cout << "-----------------------------------" << std::endl;
    // --- END DEBUG ---


    // buffer for compressed data (now includes space for the Gap Array)
    std::unique_ptr<UNIT_TYPE[]> compressed
        = std::make_unique<UNIT_TYPE[]>(total_compressed_units);

    // 3. Encode the main data.
    // ... in src/demo.cc

        // 3. Encode the main data.
        TIMER_START(timings, "encoding")
            // CRITICAL: Call the NEW Gap Array aware encoder function
            llhuff::LLHuffmanEncoder::encode_memory_gap_array(
                compressed.get(),
                total_compressed_units,
                buffer.data(), size, gap_array_size_units, enc_table);
        TIMER_STOP
        
        // ... rest of demo.cc
            
    // --- DEBUG: Print Gap Array Start Points (First 5 raw units) ---
    // This assumes the Gap Array is NOT bit-packed for this debug print
    // The Gap Array holds the absolute bit offset for each segment.
    UNIT_TYPE* gap_array_ptr_host = compressed.get();
    std::cout << "\n--- DEBUG: Raw Gap Array Units (First 5) ---" << std::endl;
    for (size_t i = 0; i < std::min(gap_array_size_units, (size_t)5); ++i) {
        // Print as 64-bit if UNIT_TYPE is 32-bit and gap offsets are 64-bit to be safe
        std::cout << "Raw Gap Array Unit [" << i << "]: " 
                  << (unsigned long long)gap_array_ptr_host[i] << std::endl; 
    }
    std::cout << "----------------------------------------------" << std::endl;
    // --- END DEBUG ---

    // --- GAP ARRAY IMPLEMENTATION CHANGES END HERE (HOST) ---

    // select CUDA device
    cudaSetDevice(compute_device_id);
    CUERR

    // define input and output buffers. 
    // total_compressed_units * sizeof(UNIT_TYPE) is the total size (Gap Array + Data)
    auto in_buf = std::make_shared<cuhd::CUHDInputBuffer>(
        reinterpret_cast<std::uint8_t*> (compressed.get()),
        total_compressed_units * sizeof(UNIT_TYPE));
        
    auto out_buf = std::make_shared<cuhd::CUHDOutputBuffer>(size);

    auto gpu_in_buf = std::make_shared<cuhd::CUHDGPUInputBuffer>(in_buf);
    auto gpu_table = std::make_shared<cuhd::CUHDGPUCodetable>(dec_table);
    auto gpu_out_buf = std::make_shared<cuhd::CUHDGPUOutputBuffer>(out_buf);

    // auxiliary memory for decoding
    // The decoder memory needs to know the total size of the input buffer.
    auto gpu_decoder_memory = std::make_shared<cuhd::CUHDGPUDecoderMemory>(
        in_buf->get_compressed_size_units(), // Total size, including gap array
        SUBSEQ_SIZE, NUM_THREADS);
    
    // allocate gpu input and output buffers
    TIMER_START(timings, "GPU buffer allocation")
        gpu_in_buf->allocate();
        gpu_out_buf->allocate();
        gpu_table->allocate();
    TIMER_STOP

    // allocate auxiliary memory for decoding (sync info, etc.)
    TIMER_START(timings, "GPU/Host aux memory allocation")
        gpu_decoder_memory->allocate();
    TIMER_STOP
    
    // copy some data
    gpu_table->cpy_host_to_device();
    
    TIMER_START(timings, "GPU memcpy HtD (Input & Tables)")
        // Transfer the entire buffer (Gap Array + Compressed Data)
        gpu_in_buf->cpy_host_to_device(); 
        
        // Assuming host encoder stored the symbol counts in the aux memory on host:
        // gpu_decoder_memory->cpy_host_to_device(); // <-- Uncomment if your aux memory contains symbol counts
        
    TIMER_STOP
    
    // decode
        TIMER_START(timings, "decoding")
        cuhd::CUHDGPUDecoder::decode(
            gpu_in_buf,
            total_compressed_units,
            gpu_out_buf, out_buf->get_uncompressed_size(),
            gpu_table, gpu_decoder_memory,
            MAX_CODEWORD_LENGTH,
            SUBSEQ_SIZE,
            NUM_THREADS,
            // CRITICAL FIX: Add the new start unit index argument
            compressed_data_start_unit, // New argument: Index to skip Gap Array
            gap_element_num,            // 11th argument
            gap_array_size_units        // 12th argument
        );
    TIMER_STOP
    
    // copy decoded data back to host
    TIMER_START(timings, "GPU memcpy DtH")
        gpu_out_buf->cpy_device_to_host();
    TIMER_STOP;

    // print timings
    for(auto &i: timings) {
        std::cout << i.first << ".. " << i.second << "µs" << std::endl;
    }
    
    // compare decompressed output to uncompressed input
    cuhd::CUHDUtil::equals(buffer.data(),
        out_buf->get_decompressed_data().get(), size) ? std::cout << std::endl
                : std::cout << std::endl << "mismatch" << std::endl;
    
    return 0;
}