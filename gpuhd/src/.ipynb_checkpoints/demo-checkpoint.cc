#include <iostream>
#include <algorithm>
#include <random>
#include <fstream> // ** ADDED: Required for file I/O **
#include <limits>  // ** ADDED: Required for std::numeric_limits **
#include <string>  // ** ADDED: Required for string manipulation **

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
#define INPUT_FILENAME "../data100_100.bin"

int main(int argc, char** argv) {
    // name of the binary file
    const char* bin = argv[0];
    
    // compute device to use
    const std::int64_t compute_device_id = atoi(argv[1]);
    
    // input size in MB (used as the default size if no file is present)
    const long int cli_size_bytes = atoi(argv[2]) * 1024 * 1024;
    long int size = cli_size_bytes; // Actual size, may be overwritten by file size
    
    if(compute_device_id < 0 || size < 1) {
        std::cout << "USAGE: " << bin << " <compute device index> "
        << "<size of input in megabytes>" << std::endl;
        return 1;
    }

    // vector for storing time measurements
    std::vector<std::pair<std::string, size_t>> timings;
    
    // generate random data for testing
    std::vector<SYMBOL_TYPE> buffer;

    // ** START MODIFIED DATA LOADING/GENERATION BLOCK **
    
    std::ifstream infile(INPUT_FILENAME, std::ios::binary | std::ios::ate);

    if (infile.is_open()) {
        // File exists: Read data and update size
        size = infile.tellg();
        infile.seekg(0, std::ios::beg);
        buffer.resize(size);
        infile.read(reinterpret_cast<char*>(buffer.data()), size);
        infile.close();
        std::cout << "Input source: Read " << (size / (1024.0 * 1024.0)) << " MB from file: " << INPUT_FILENAME << std::endl;

        // NOTE: The original TIMER_START/STOP for data generation is skipped here.
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
    // ** END MODIFIED DATA LOADING/GENERATION BLOCK **

    std::shared_ptr<std::vector<llhuff::LLHuffmanEncoder::Symbol>> lengths;
    std::shared_ptr<llhuff::LLHuffmanEncoderTable> enc_table;
    std::shared_ptr<cuhd::CUHDCodetable> dec_table;

    std::unique_ptr<UNIT_TYPE[]> compressed;
    
    TIMER_START(timings, "Compression time")
    
        // Determine optimal code lengths
        lengths = llhuff::LLHuffmanEncoder::get_symbol_lengths(
            buffer.data(), size);
    
        // Generate encoder table (also computes compressed size)
        enc_table = llhuff::LLHuffmanEncoder::get_encoder_table(lengths);
    
        // Generate decoder table (required for later decompression)
        dec_table = llhuff::LLHuffmanEncoder::get_decoder_table(enc_table);
    
        // Allocate buffer for compressed output
        compressed = std::make_unique<UNIT_TYPE[]>(enc_table->compressed_size);
    
        // Encode input data
        llhuff::LLHuffmanEncoder::encode_memory(
            compressed.get(),
            enc_table->compressed_size,
            buffer.data(),
            size,
            enc_table
        );
    
    TIMER_STOP

    //VERBOSE TIMING INFORMATION
    /*
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
    */
    
        // select CUDA device
    	cudaSetDevice(compute_device_id);
    	CUERR
    
    	// define input and output buffers
        auto in_buf = std::make_shared<cuhd::CUHDInputBuffer>(
            reinterpret_cast<std::uint8_t*> (compressed.get()),
            enc_table->compressed_size * sizeof(UNIT_TYPE));
            
        auto out_buf = std::make_shared<cuhd::CUHDOutputBuffer>(size);
    
        auto gpu_in_buf = std::make_shared<cuhd::CUHDGPUInputBuffer>(in_buf);
        auto gpu_table = std::make_shared<cuhd::CUHDGPUCodetable>(dec_table);
        auto gpu_out_buf = std::make_shared<cuhd::CUHDGPUOutputBuffer>(out_buf);
    
        // auxiliary memory for decoding
        auto gpu_decoder_memory = std::make_shared<cuhd::CUHDGPUDecoderMemory>(
            in_buf->get_compressed_size_units(),
            SUBSEQ_SIZE, NUM_THREADS);
     
    TIMER_START(timings, "Decompression time")
    
        // Allocate GPU buffers
        gpu_in_buf->allocate();
        gpu_out_buf->allocate();
        gpu_table->allocate();
    
        // Allocate auxiliary decoder memory
        gpu_decoder_memory->allocate();
    
        // Copy decoder tables (small, but required)
        gpu_table->cpy_host_to_device();
    
        // Copy compressed input (HtD)
        gpu_in_buf->cpy_host_to_device();
    
        // Decode kernel
        cuhd::CUHDGPUDecoder::decode(
            gpu_in_buf, in_buf->get_compressed_size_units(),
            gpu_out_buf, out_buf->get_uncompressed_size(),
            gpu_table, gpu_decoder_memory,
            MAX_CODEWORD_LENGTH, SUBSEQ_SIZE, NUM_THREADS
        );
    
        // Copy decompressed output back (DtH)
        gpu_out_buf->cpy_device_to_host();
    
    TIMER_STOP

    //VERBOSE TIMING INFORMATION
    /*
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
    
    TIMER_START(timings, "GPU memcpy HtD")
        gpu_in_buf->cpy_host_to_device();
    TIMER_STOP
    
    // decode
    TIMER_START(timings, "decoding")
        cuhd::CUHDGPUDecoder::decode(
            gpu_in_buf, in_buf->get_compressed_size_units(),
            gpu_out_buf, out_buf->get_uncompressed_size(),
            gpu_table, gpu_decoder_memory,
            MAX_CODEWORD_LENGTH, SUBSEQ_SIZE, NUM_THREADS);
    TIMER_STOP
    
    // copy decoded data back to host
    TIMER_START(timings, "GPU memcpy DtH")
        gpu_out_buf->cpy_device_to_host();
    TIMER_STOP;
    */

    // print timings
    for(auto &i: timings) {
        std::cout << i.first << ".. " << i.second << "µs" << std::endl;
    }

    //throughputs
    std::cout << "Compression throughput (mb/s): " << size/timings[0].second << "\n";
    std::cout << "Decompression throughput (mb/s): " << size/timings[1].second << "\n";
    
    
    // compare decompressed output to uncompressed input
    cuhd::CUHDUtil::equals(buffer.data(),
        out_buf->get_decompressed_data().get(), size) ? std::cout << std::endl
            : std::cout << std::endl << "mismatch" << std::endl;
    
    return 0;
}