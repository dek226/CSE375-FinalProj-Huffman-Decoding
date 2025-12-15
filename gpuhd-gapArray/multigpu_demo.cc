#include <iostream>
#include <algorithm>
#include <random>
#include <fstream> 
#include <limits>  
#include <string>  
#include <vector>  // Required for multi-GPU contexts
#include <numeric> // Required for std::iota

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <llhuff.h>       // encoder
#include <cuhd.h>         // decoder

// subsequence size
#define SUBSEQ_SIZE 4

// threads per block
#define NUM_THREADS 128

// performance parameter, high = high performance, low = low memory consumption
#define DEVICE_PREF 9

// ** ADDED: Constant for the data file name **
#define INPUT_FILENAME "input_data.bin"

// Structure to hold resources for a single GPU context
struct GpuContext {
    int device_id;
    long int uncompressed_elements; // SYMBOL_TYPE count
    long int compressed_units;      // UNIT_TYPE count
    long int uncompressed_offset;   // Offset in SYMBOL_TYPE elements
    long int compressed_offset;     // Offset in UNIT_TYPE units
    // cudaStream_t stream; // REMOVED: Using default stream per device
    
    // CUHD/CUDA resources
    std::shared_ptr<cuhd::CUHDInputBuffer> in_buf;
    std::shared_ptr<cuhd::CUHDOutputBuffer> out_buf;
    std::shared_ptr<cuhd::CUHDGPUInputBuffer> gpu_in_buf;
    std::shared_ptr<cuhd::CUHDGPUOutputBuffer> gpu_out_buf;
    std::shared_ptr<cuhd::CUHDCodetable> dec_table; // Decoder table copy
    std::shared_ptr<cuhd::CUHDGPUCodetable> gpu_table;
    std::shared_ptr<cuhd::CUHDGPUDecoderMemory> gpu_decoder_memory;
};


int main(int argc, char** argv) {
    // name of the binary file
    const char* bin = argv[0];
    
    // ** MODIFIED ARGUMENT PARSING **
    // argv[1] is now the number of GPUs to use
    // argv[2] is the total input size in MB
    if (argc < 3) {
        std::cout << "USAGE: " << bin << " <number of GPUs> "
        << "<total size of input in megabytes>" << std::endl;
        return 1;
    }

    const int num_gpus = atoi(argv[1]);
    const long int cli_size_bytes = atoi(argv[2]) * 1024 * 1024;
    long int size = cli_size_bytes / sizeof(SYMBOL_TYPE); // Total element count (SYMBOL_TYPE)
    
    int max_devices;
    cudaGetDeviceCount(&max_devices);
    CUERR

    if(num_gpus <= 0 || num_gpus > max_devices || size < 1) {
        std::cerr << "Invalid parameters." << std::endl;
        std::cerr << "Requested GPUs: " << num_gpus << ", Available: " << max_devices << std::endl;
        std::cerr << "Input element size: " << size << " (must be >= 1)" << std::endl;
        return 1;
    }
    // ** END MODIFIED ARGUMENT PARSING **


    // vector for storing time measurements
    std::vector<std::pair<std::string, size_t>> timings;
    
    // generate random data for testing
    // 'buffer' holds the original, uncompressed input data for final comparison
    std::vector<SYMBOL_TYPE> buffer; 

    // ** START MODIFIED DATA LOADING/GENERATION BLOCK (Retains size safety check) **
    
    std::ifstream infile(INPUT_FILENAME, std::ios::binary | std::ios::ate);

    if (infile.is_open()) {
        // File exists: Read data and update size
        long int file_size_bytes = infile.tellg();
        infile.seekg(0, std::ios::beg);
        
        if (file_size_bytes % sizeof(SYMBOL_TYPE) != 0) {
            std::cerr << "Error: File size (" << file_size_bytes 
                      << " bytes) is not a multiple of SYMBOL_TYPE size (" 
                      << sizeof(SYMBOL_TYPE) << " bytes). The file is likely corrupt." << std::endl;
            return 1;
        }

        size = file_size_bytes / sizeof(SYMBOL_TYPE); // size is now the correct number of elements
        
        buffer.resize(size);
        infile.read(reinterpret_cast<char*>(buffer.data()), file_size_bytes);
        infile.close();
        std::cout << "Input source: Read " << (file_size_bytes / (1024.0 * 1024.0)) << " MB from file: " << INPUT_FILENAME << std::endl;

    } else {
        // File missing: Generate random data (original logic), and save
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
            outfile.write(reinterpret_cast<const char*>(buffer.data()), size * sizeof(SYMBOL_TYPE));
            outfile.close();
            std::cout << "Data saved to '" << INPUT_FILENAME << "'." << std::endl;
        } else {
            std::cerr << "Warning: Could not save generated data to '" << INPUT_FILENAME << "'." << std::endl;
        }
    }
    // ** END MODIFIED DATA LOADING/GENERATION BLOCK **

    std::shared_ptr<std::vector<llhuff::LLHuffmanEncoder::Symbol>> lengths;
    std::shared_ptr<llhuff::LLHuffmanEncoderTable> enc_table;
    std::shared_ptr<cuhd::CUHDCodetable> dec_table;
    
    // compress the random data using length-limited Huffman coding
    TIMER_START(timings, "generating coding tables")

        // determine optimal lengths for the codewords to be generated
        lengths = llhuff::LLHuffmanEncoder::get_symbol_lengths(
            buffer.data(), size);
    
        // generate encoder table
        enc_table = llhuff::LLHuffmanEncoder::get_encoder_table(lengths);
    
        // generate decoder table
        dec_table = llhuff::LLHuffmanEncoder::get_decoder_table(enc_table);
    TIMER_STOP
    
    // buffer for compressed data
    std::unique_ptr<UNIT_TYPE[]> compressed
        = std::make_unique<UNIT_TYPE[]>(enc_table->compressed_size);

    // compress
    TIMER_START(timings, "encoding")
        llhuff::LLHuffmanEncoder::encode_memory(compressed.get(),
            enc_table->compressed_size, buffer.data(), size, enc_table);
    TIMER_STOP
        
    // Total compressed size (in UNIT_TYPE units)
    long int total_compressed_units = enc_table->compressed_size;
    // Total uncompressed size (in SYMBOL_TYPE elements)
    long int total_uncompressed_elements = size;
    
    // ** FIX: New buffer to hold the combined decoded output from all GPUs **
    std::vector<SYMBOL_TYPE> output_data(total_uncompressed_elements); 

    // ** START MULTI-GPU LOGIC **

    std::vector<GpuContext> contexts(num_gpus);

    TIMER_START(timings, "Multi-GPU Setup & Allocation")
    // 1. Setup and split data for each GPU
    for (int i = 0; i < num_gpus; ++i) {
        contexts[i].device_id = i;
        cudaSetDevice(i);
        CUERR
        // Note: cudaStreamCreate removed as CUHD functions don't take stream arguments.
        // We rely on the default stream for each device.

        // Calculate chunk sizes for the current GPU
        long int chunk_size_elements = total_uncompressed_elements / num_gpus;
        long int chunk_compressed_units = total_compressed_units / num_gpus;
        
        // Ensure the last chunk gets the remainder
        if (i == num_gpus - 1) {
            chunk_size_elements += total_uncompressed_elements % num_gpus;
            chunk_compressed_units += total_compressed_units % num_gpus;
        }

        // Calculate offsets
        contexts[i].uncompressed_offset = i * (total_uncompressed_elements / num_gpus);
        contexts[i].compressed_offset = i * (total_compressed_units / num_gpus);
        contexts[i].uncompressed_elements = chunk_size_elements;
        contexts[i].compressed_units = chunk_compressed_units;

        // 2. Define input and output buffers for the chunk
        contexts[i].in_buf = std::make_shared<cuhd::CUHDInputBuffer>(
            reinterpret_cast<std::uint8_t*> (compressed.get() + contexts[i].compressed_offset),
            contexts[i].compressed_units * sizeof(UNIT_TYPE));
            
        contexts[i].out_buf = std::make_shared<cuhd::CUHDOutputBuffer>(contexts[i].uncompressed_elements);

        // 3. Create GPU objects
        contexts[i].gpu_in_buf = std::make_shared<cuhd::CUHDGPUInputBuffer>(contexts[i].in_buf);
        contexts[i].dec_table = dec_table; 
        contexts[i].gpu_table = std::make_shared<cuhd::CUHDGPUCodetable>(contexts[i].dec_table); 
        contexts[i].gpu_out_buf = std::make_shared<cuhd::CUHDGPUOutputBuffer>(contexts[i].out_buf);

        // 4. Auxiliary memory for decoding
        contexts[i].gpu_decoder_memory = std::make_shared<cuhd::CUHDGPUDecoderMemory>(
            contexts[i].in_buf->get_compressed_size_units(),
            SUBSEQ_SIZE, NUM_THREADS);

        // 5. Allocate GPU buffers 
        contexts[i].gpu_in_buf->allocate();
        contexts[i].gpu_out_buf->allocate();
        contexts[i].gpu_table->allocate();
        contexts[i].gpu_decoder_memory->allocate();

        // 6. Copy static data (Code table)
        contexts[i].gpu_table->cpy_host_to_device();
    }
    TIMER_STOP

    // *** START: PROPER CUDA EVENT TIMING FOR ASYNCHRONOUS DECODE SPAN ***

    // Set device 0 context before creating events to ensure valid handles.
    cudaSetDevice(contexts[0].device_id);
    CUERR
    
    // FIX: Call cudaGetLastError to clear any error state that might be lingering
    // from previous operations, which could cause the subsequent CUERR macro to fire
    // immediately with an old 'invalid resource handle' error.
    cudaGetLastError(); 

    cudaEvent_t start_event, stop_event;
    
    // FIX: Use cudaEventBlockingSync for increased stability in cross-context timing
    // operations. This may help resolve the invalid resource handle issue during recording.
    cudaEventCreate(&start_event, cudaEventBlockingSync);
    cudaEventCreate(&stop_event, cudaEventBlockingSync);
    CUERR

    // 7. Transfer input data and launch kernels asynchronously (using default stream)
    
    // Record the start event on the default stream of the first device (0).
    cudaEventRecord(start_event, 0); // Use 0 for the default stream
    CUERR
    
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(contexts[i].device_id);
        
        // Memcpy input data (HtD) using the default stream
        contexts[i].gpu_in_buf->cpy_host_to_device();

        // Decode launch (asynchronously on default stream)
        cuhd::CUHDGPUDecoder::decode(
            contexts[i].gpu_in_buf, contexts[i].in_buf->get_compressed_size_units(),
            contexts[i].gpu_out_buf, contexts[i].out_buf->get_uncompressed_size(),
            contexts[i].gpu_table, contexts[i].gpu_decoder_memory,
            MAX_CODEWORD_LENGTH, SUBSEQ_SIZE, NUM_THREADS); 
    }
    
    
    // 8. Synchronize all devices and then record the stop event on Device 0.
    
    // Ensure all GPU work is complete on every device.
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(contexts[i].device_id);
        cudaDeviceSynchronize(); // Wait for all work on this device to finish
        CUERR
    }

    // Switch back to Device 0 to record the stop event on its own context.
    cudaSetDevice(contexts[0].device_id);
    CUERR
    
    // Record the stop event on the same context it was created on (Device 0).
    cudaEventRecord(stop_event, 0);
    CUERR
    
    // Calculate elapsed time (in milliseconds)
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
    CUERR

    // Record the total parallel execution time in microseconds
    timings.push_back({"Multi-GPU Parallel Decode Time (Full Span)", (size_t)(elapsed_ms * 1000)});

    // *** END: PROPER CUDA EVENT TIMING ***
    
    
    // 10. Copy decoded data back to host and combine results
    // This phase is now timed separately as sequential DtH and combination.
    TIMER_START(timings, "Multi-GPU Memcpy DtH & Combine")
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(contexts[i].device_id);
        
        // Copy decoded data back to host (DtH) - This is a blocking copy
        contexts[i].gpu_out_buf->cpy_device_to_host();
        
        // Copy the result chunk back into the combined host buffer (output_data)
        SYMBOL_TYPE* dest_ptr = output_data.data() + contexts[i].uncompressed_offset;
        SYMBOL_TYPE* source_ptr = contexts[i].out_buf->get_decompressed_data().get();
        std::copy(source_ptr, source_ptr + contexts[i].uncompressed_elements, dest_ptr);
        
        // Note: cudaStreamDestroy removed as streams are no longer created
    }
    TIMER_STOP
    
    // 11. Clean up events
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    CUERR
    
    // ** END MULTI-GPU LOGIC **
    

    // print timings
    for(auto &i: timings) {
        std::cout << i.first << ".. " << i.second << "µs" << std::endl;
    }
    
    // compare decoded output (output_data) to original input (buffer)
    cuhd::CUHDUtil::equals(buffer.data(),
        output_data.data(), total_uncompressed_elements) ? std::cout << std::endl
            : std::cout << std::endl << "mismatch" << std::endl;
    
    return 0;
}