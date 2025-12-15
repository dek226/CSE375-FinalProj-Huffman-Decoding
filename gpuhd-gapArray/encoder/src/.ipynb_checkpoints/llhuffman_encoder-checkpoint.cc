/*****************************************************************************
 *
 * LLHuff - Length-limited Huffman encoding
 *
 * released under LGPL-3.0
 *
 * 2017-2018 André Weißenberger
 *
 *****************************************************************************/

#include "llhuffman_encoder.h"

#include <map>
#include <unordered_map>
#include <algorithm>
#include <cmath>

std::shared_ptr<std::vector<llhuff::LLHuffmanEncoder::Symbol>>
    llhuff::LLHuffmanEncoder::get_symbol_lengths(
    SYMBOL_TYPE* data, size_t size) {

    // collect source symbol frequencies
    std::unordered_map<SYMBOL_TYPE, size_t> freq;
    
    for(size_t i = 0; i < size; ++i)
        ++freq[data[i]];

    // if number of symbols exceeds maximum number of leaf nodes,
    // cancel and return nullptr
    if(log2(freq.size()) > MAX_CODEWORD_LENGTH) {
        return nullptr;
    }
    
    // sort symbols by frequencies in increasing order
    std::vector<std::pair<SYMBOL_TYPE, size_t>>
        symbols(freq.begin(), freq.end());

    if(symbols.size() == 1) {

        // input consists of only one symbol
        Symbol s; s.length = 1; s.symbol = symbols.at(0).first;
        std::vector<Symbol> vec(1);
        vec.at(0) = s;

        return std::make_shared<std::vector<Symbol>>(vec);
    }

    std::sort(symbols.begin(), symbols.end(),
        [](std::pair<SYMBOL_TYPE, size_t> &a,
            std::pair<SYMBOL_TYPE, size_t> &b) -> bool
            {return a.second < b.second;});
    
    // create lists of coins (denomination, numismatic value)
    std::vector<std::vector<CoinPackage>> denom_lists;
    denom_lists.resize(MAX_CODEWORD_LENGTH);

    // create coins
    for(size_t i = 0; i < MAX_CODEWORD_LENGTH; ++i) {
        size_t second_counter = 0;

        std::vector<CoinPackage> denom;
        denom.resize(symbols.size());

        for(size_t j = 0; j < symbols.size(); ++j) {
            CoinPackage coin;

            const float denomination = 1.0f /
                (float) (1 << (MAX_CODEWORD_LENGTH - i));
            const float numismatic = (float) (symbols[j].second)
                / (float) (size);
            const SYMBOL_TYPE sym = symbols[j].first;

            coin.denomination = denomination;
            coin.numismatic = numismatic;
            
            coin.coins.push_back(Coin{denomination, numismatic, sym});

            denom.at(second_counter) = coin;
            ++second_counter;
        }
        
        denom_lists.at(i) = denom;
    }
    
    // apply package / merge algorithm
    for(size_t i = 0; i < MAX_CODEWORD_LENGTH - 1; ++i) {
        std::vector<CoinPackage> new_row;

        // package
        if(denom_lists.at(i).size() % 2 != 0) {
            denom_lists.at(i).pop_back();
        }
        
        for(size_t j = 0; j <= (denom_lists.at(i).size() - 2); ++++j) {
            CoinPackage new_coin = denom_lists.at(i).at(j).package(
                &(denom_lists.at(i).at(j + 1)));

            new_row.push_back(new_coin);
        }
        
        std::vector<CoinPackage> dst;

        // merge
        std::merge(
            new_row.begin(), new_row.end(),
            denom_lists.at(i + 1).begin(), denom_lists.at(i + 1).end(),
            std::back_inserter(dst),
            [](CoinPackage &a, CoinPackage &b) -> bool {
                return a.numismatic <= b.numismatic;
            });
        
        denom_lists.at(i + 1) = dst;
    }

    // keep coin packages whose denominations add up to symbols.size() - 1,
    // delete the remaining coins
    denom_lists.back().resize(2 * (symbols.size() - 1));

    // merge all coins into a vector
    std::vector<Coin> coins;

    for(size_t i = 0; i < denom_lists.back().size(); ++i) {

        coins.reserve(coins.size() + denom_lists.back().at(i).coins.size());

        coins.insert(coins.end(), denom_lists.back().at(i).coins.begin(),
            denom_lists.back().at(i).coins.end());
    }
    
    // count coins with equal symbol
    std::unordered_map<SYMBOL_TYPE, size_t> num_coins_of_symbol;

    for(size_t i = 0; i < coins.size(); ++i)
        ++num_coins_of_symbol[coins[i].symbol];
    
    // assign codeword length to symbols
    std::shared_ptr<std::vector<Symbol>> symbol_length
        = std::make_shared<std::vector<Symbol>>();
    
    symbol_length->resize(num_coins_of_symbol.size());

    size_t second_counter = 0;
    for(auto &i: num_coins_of_symbol) {

        symbol_length->at(second_counter).symbol = i.first;
        symbol_length->at(second_counter).length = i.second;
        symbol_length->at(second_counter).count = freq[i.first];

        ++second_counter;
    }

    // sort symbols by increasing codeword length
    std::sort(symbol_length->begin(), symbol_length->end(),
        [](Symbol &a, Symbol &b) -> bool
            {return a.length < b.length;});
    
    return symbol_length;
}

// --- New Function to Implement Gap Array Encoding ---

// NOTE: This assumes SEGMENT_SIZE_BITS is defined somewhere (like 8192)
#define SEGMENT_SIZE_BITS 8192
#define UNIT_SIZE_BITS (sizeof(UNIT_TYPE) * 8)

void llhuff::LLHuffmanEncoder::encode_memory_gap_array(
    UNIT_TYPE* out, size_t size_out, // Total buffer size
    SYMBOL_TYPE* in, size_t size_in,
    size_t gap_array_size_units, // Size of the Gap Array in UNIT_TYPEs
    std::shared_ptr<llhuff::LLHuffmanEncoderTable> encoder_table) {

    // 1. Set up pointers to the two distinct sections in the buffer
    UNIT_TYPE* gap_array_ptr = out;
    UNIT_TYPE* compressed_data_ptr = out + gap_array_size_units;

    // 2. Sequential Encoding variables
    UNIT_TYPE window = 0;
    size_t at = 0;
    size_t in_unit = 0;
    
    // 3. Gap Array Tracking variables
    size_t current_compressed_bit_offset = 0;
    size_t current_unit_index = 0; // Index into the compressed_data_ptr
    size_t current_segment_bit_size = 0;
    size_t current_segment_index = 0;
    
    // Write the starting offset for segment 0 (which is 0)
    if (current_segment_index < gap_array_size_units) {
         gap_array_ptr[current_segment_index] = 0;
         current_segment_index++;
    }

    // Iterate over the output UNIT_TYPEs (compressed data units only)
    size_t max_compressed_units = size_out - gap_array_size_units;
    for(current_unit_index = 0; current_unit_index < max_compressed_units && in_unit < size_in; ++current_unit_index) {

        // --- Loop to fill the current output unit (bit packing) ---
        // This logic is mostly copied from the original encode_memory
        
        // Loop 1: While we can pack full codewords into the window
        auto code = encoder_table->dict[in[in_unit]];
        
        while(at + code.length < UNIT_SIZE_BITS && in_unit < size_in) {
            
            // Check for segment boundary *before* writing the bit

            // CRITICAL: If the next codeword pushes us past the segment size,
            // we must record the offset for the next segment's start.
            if (current_segment_bit_size + code.length >= SEGMENT_SIZE_BITS) {
                 // Calculate the total bit offset for the next segment start
                 current_compressed_bit_offset += current_segment_bit_size;

                 // Record the start bit offset for the *next* segment
                 if (current_segment_index < gap_array_size_units) {
                    gap_array_ptr[current_segment_index] = (UNIT_TYPE)current_compressed_bit_offset;
                    current_segment_index++;
                 }
                 
                 // Reset the segment size, accounting for overflow of the current codeword
                 current_segment_bit_size = (current_segment_bit_size + code.length) - SEGMENT_SIZE_BITS;
                 
            } else {
                 current_segment_bit_size += code.length;
            }
            
            window <<= code.length;
            window += code.codeword;
            at += code.length;
            ++in_unit;

            if(in_unit < size_in)
                code = encoder_table->dict[in[in_unit]];
        }
        
        // Loop 2: Write the final partial codeword and flush the unit
        const size_t diff = at + code.length - UNIT_SIZE_BITS; // Bits overflowing to next unit

        // Update tracking for the partial codeword
        if (current_segment_bit_size + code.length >= SEGMENT_SIZE_BITS) {
             current_compressed_bit_offset += current_segment_bit_size;
             if (current_segment_index < gap_array_size_units) {
                gap_array_ptr[current_segment_index] = (UNIT_TYPE)current_compressed_bit_offset;
                current_segment_index++;
             }
             current_segment_bit_size = (current_segment_bit_size + code.length) - SEGMENT_SIZE_BITS;
        } else {
             current_segment_bit_size += code.length;
        }

        window <<= code.length - diff;
        window += (code.codeword >> diff);
        
        // Write the fully packed UNIT_TYPE to the data section
        compressed_data_ptr[current_unit_index] = window;
        
        // Prepare for next unit
        window = code.codeword & ~(~0 << diff);
        at = diff;

        ++in_unit;
    }
    
    // --- Post-loop cleanup (write the last unit if not empty) ---
    if (at > 0) {
        // Shift window to MSB and write the last unit
        compressed_data_ptr[current_unit_index] = window << (UNIT_SIZE_BITS - at);
    }
}

std::shared_ptr<llhuff::LLHuffmanEncoderTable>
    llhuff::LLHuffmanEncoder::get_encoder_table(
    std::shared_ptr<std::vector<Symbol>> symbol_lengths) {

    llhuff::LLHuffmanEncoderTable table;

    // calculate total size of compressed output in bytes
    size_t size = 0;

    for(size_t i = 0; i < symbol_lengths->size(); ++i)
        size += symbol_lengths->at(i).length * symbol_lengths->at(i).count;

    // ceil to integer byte count
    if(size % 8 != 0)
        size += 8 - (size % 8);

    size /= 8;

    // convert to unit size
    table.compressed_size = size % sizeof(UNIT_TYPE) == 0 ?
        size / sizeof(UNIT_TYPE) : (size / sizeof(UNIT_TYPE)) + 1;

    // generate canonical codewords
    UNIT_TYPE code = 0;
    size_t current_length = symbol_lengths->at(0).length;
    
    for(size_t i = 0; i < symbol_lengths->size(); ++i) {
        table.dict[symbol_lengths->at(i).symbol] = {code, current_length};

        size_t next_length = (i + 1 < symbol_lengths->size())
            ? symbol_lengths->at(i + 1).length : current_length;

        code = (code + 1) << (next_length - current_length);

        current_length = next_length;
    }
    
    return std::make_shared<llhuff::LLHuffmanEncoderTable>(table);
}

void llhuff::LLHuffmanEncoder::encode_memory(UNIT_TYPE* out, size_t size_out,
    SYMBOL_TYPE* in, size_t size_in,
    std::shared_ptr<llhuff::LLHuffmanEncoderTable> encoder_table) {

    UNIT_TYPE* out_ptr = out;
         
    const size_t max_bits = sizeof(UNIT_TYPE) * 8;

    UNIT_TYPE window = 0;
    
    size_t at = 0;
    size_t in_unit = 0;

    for(size_t i = 0; i < size_out && in_unit < size_in; ++i) {
        auto code = encoder_table->dict[in[in_unit]];
        
        while(at + code.length < max_bits && in_unit < size_in) {
            window <<= code.length;
            window += code.codeword;
            at += code.length;
            ++in_unit;

            if(in_unit < size_in)
                code = encoder_table->dict[in[in_unit]];
        }
        
        const size_t diff = at + code.length - max_bits;

        window <<= code.length - diff;
        window += (code.codeword >> diff);
        
        out_ptr[i] = window;
        
        window = code.codeword & ~(~0 << diff);
        at = diff;

        ++in_unit;
    }
}

std::shared_ptr<cuhd::CUHDCodetable>
    llhuff::LLHuffmanEncoder::get_decoder_table(
    std::shared_ptr<llhuff::LLHuffmanEncoderTable> enc_table) {

    std::shared_ptr<cuhd::CUHDCodetable> table
        = std::make_shared<cuhd::CUHDCodetable>(enc_table->dict.size());

    cuhd::CUHDCodetableItemSingle* dec_ptr = table->get();
    
    for(auto &i: enc_table->dict) {    
        const SYMBOL_TYPE symbol = i.first;
        const UNIT_TYPE codeword = i.second.codeword;
        const BIT_COUNT_TYPE length = i.second.length;

        const size_t shift = MAX_CODEWORD_LENGTH - length;
        const size_t num_entries = 1 << shift;
        
        for(size_t j = 0; j < num_entries; ++j)
            dec_ptr[(codeword << shift) + j] = {length, symbol};
    }
    
    return table;
}

