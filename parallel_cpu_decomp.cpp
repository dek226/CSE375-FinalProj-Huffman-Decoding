// parallel_huffman_fixed.cpp
// Improved Huffman encode/decode with parallel decode boundary-fix + small LUT
// Compile: g++ -O3 -fopenmp -std=c++17 parallel_huffman_fixed.cpp -o parallel_decomp_cpu

#include <vector>
#include <unordered_map>
#include <queue>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <omp.h>
#include <array>

int thread_count = 4;
constexpr size_t DECODE_BLOCK_SIZE = 1024 * 64; // 64KB block size
constexpr int LUT_BITS = 8; // small byte-wise LUT

class HuffmanParallelCPU {
public:
    std::vector<uint8_t> encode(const std::vector<uint8_t>& data) {
        if (data.empty()) return {};

        auto freq = countFrequenciesParallel(data);
        Node* root = buildTree(freq);
        std::unordered_map<uint8_t, std::string> codes;
        buildCodes(root, "", codes);

        // find max code length for decoding alignment use
        max_code_len_ = 0;
        for (auto &p : codes) if (p.second.size() > max_code_len_) max_code_len_ = p.second.size();

        std::string bitstream = encodeParallel(data, codes);
        int padding = (8 - (bitstream.size() % 8)) % 8;
        bitstream.append(padding, '0');

        std::vector<uint8_t> output;
        writeHeader(output, padding, codes);
        for (size_t i = 0; i < bitstream.size(); i += 8) {
            uint8_t byte = static_cast<uint8_t>(std::stoi(bitstream.substr(i, 8), nullptr, 2));
            output.push_back(byte);
        }

        freeTree(root);
        return output;
    }

    std::vector<uint8_t> decode(const std::vector<uint8_t>& compressed) {
        return decodeParallel(compressed);
    }

private:
    struct Node {
        uint8_t symbol;
        int freq;
        Node* left;
        Node* right;
        bool isLeaf;
        Node(uint8_t s, int f) : symbol(s), freq(f), left(nullptr), right(nullptr), isLeaf(true) {}
        Node(Node* l, Node* r) : symbol(0), freq(l->freq + r->freq), left(l), right(r), isLeaf(false) {}
    };
    struct NodeCompare { bool operator()(Node* a, Node* b) const { return a->freq > b->freq; } };

    // ---------------- ENCODE HELPERS ----------------
    std::unordered_map<uint8_t,int> countFrequenciesParallel(const std::vector<uint8_t>& data) {
        constexpr int SYMBOLS = 256;
        std::vector<std::array<int,SYMBOLS>> local(thread_count);
        for (auto &arr : local) arr.fill(0);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& table = local[tid];
            #pragma omp for nowait
            for (size_t i = 0; i < data.size(); ++i) table[data[i]]++;
        }

        std::array<int,SYMBOLS> global{};
        for (int t=0;t<thread_count;++t) for (int s=0;s<SYMBOLS;++s) global[s] += local[t][s];

        std::unordered_map<uint8_t,int> freq;
        for (int s=0;s<SYMBOLS;++s) if (global[s] > 0) freq[static_cast<uint8_t>(s)] = global[s];
        return freq;
    }

    std::string encodeParallel(const std::vector<uint8_t>& data, const std::unordered_map<uint8_t,std::string>& codes) {
        std::vector<std::string> local(thread_count);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto &out = local[tid];
            out.reserve(data.size()/thread_count * 2);
            #pragma omp for nowait
            for (size_t i=0;i<data.size();++i) out += codes.at(data[i]);
        }
        std::string result; result.reserve(data.size()*2);
        for (auto &s : local) result += s;
        return result;
    }

    // ---------------- DECODER HELPERS ----------------

    // simple binary trie node for decode
    struct TrieNode {
        int16_t symbol; // -1 = internal
        TrieNode* child[2];
        TrieNode(): symbol(-1) { child[0]=child[1]=nullptr; }
    };
    TrieNode* buildTrie(const std::unordered_map<uint8_t,std::string>& codes) {
        TrieNode* root = new TrieNode();
        for (auto &p : codes) {
            TrieNode* cur = root;
            const std::string &c = p.second;
            for (char ch : c) {
                int b = (ch == '1');
                if (!cur->child[b]) cur->child[b] = new TrieNode();
                cur = cur->child[b];
            }
            cur->symbol = static_cast<int16_t>(p.first);
        }
        return root;
    }
    void freeTrie(TrieNode* n) {
        if (!n) return;
        freeTrie(n->child[0]); freeTrie(n->child[1]);
        delete n;
    }

    // small LUT: for each byte value, if it resolves to a symbol in <= LUT_BITS bits
    struct LUTEntry { int16_t symbol; uint8_t bitsConsumed; bool valid; };
    std::array<LUTEntry, 1<<LUT_BITS> buildLUT(const std::unordered_map<uint8_t,std::string>& codes) {
        std::array<LUTEntry, 1<<LUT_BITS> lut;
        for (size_t i=0;i<lut.size();++i) lut[i] = { -1, 0, false };

        // For each code, for all possible padding suffix bits fill prefix matches
        for (auto &p : codes) {
            const std::string &c = p.second;
            if (c.size() <= LUT_BITS) {
                // build the integer for code bits; fill all suffix combos
                int prefix = 0;
                for (char ch : c) { prefix = (prefix<<1) | (ch=='1'); }
                int suffixBits = LUT_BITS - static_cast<int>(c.size());
                int base = prefix << suffixBits;
                int repeat = 1 << suffixBits;
                for (int r = 0; r < repeat; ++r) {
                    int idx = base | r;
                    lut[idx] = { static_cast<int16_t>(p.first), static_cast<uint8_t>(c.size()), true };
                }
            }
        }
        return lut;
    }

    // sequential bit-level decode within a bit-range using trie (byte-wise LUT speedup)
    std::vector<uint8_t> sequentialDecodeRange(const std::string& fullBitstream,
                                              size_t start_bit, size_t end_bit,
                                              TrieNode* trie, const std::array<LUTEntry,1<<LUT_BITS>& lut)
    {
        std::vector<uint8_t> out;
        size_t pos = start_bit;
        while (pos < end_bit) {
            // try LUT if enough bits
            if (end_bit - pos >= LUT_BITS) {
                // Read next LUT_BITS bits
                int idx = 0;
                for (int i=0;i<LUT_BITS;i++) {
                    idx = (idx<<1) | (fullBitstream[pos + i] - '0');
                }
                LUTEntry e = lut[idx];
                if (e.valid) {
                    out.push_back(static_cast<uint8_t>(e.symbol));
                    pos += e.bitsConsumed;
                    continue;
                }
            }
            
            // fallback to trie bit-by-bit
            TrieNode* cur = trie;
            size_t p = pos;
            bool found = false;
            
            // Limit trie search to avoid reading beyond end_bit (up to max_code_len_ bits)
            size_t max_search_p = std::min(pos + max_code_len_, end_bit); 

            while (p < max_search_p && cur) {
                int b = fullBitstream[p] - '0';
                cur = cur->child[b];
                p++;
                if (cur && cur->symbol >= 0) {
                    out.push_back(static_cast<uint8_t>(cur->symbol));
                    pos = p; // update main position
                    found = true;
                    break;
                }
            }
            if (!found) {
                // incomplete code at end of range; stop
                break;
            }
        }
        return out;
    }

    // Align start bit: scan backwards up to max_code_len_ bits and try to find a boundary so that the first decoded symbol ends at or after nominal_start
    size_t align_start_bit(const std::string& fullBitstream, size_t nominal_start, size_t total_bits, TrieNode* trie, const std::array<LUTEntry,1<<LUT_BITS>& lut) {
        size_t max_back = std::min(static_cast<size_t>(max_code_len_), nominal_start);
        
        // Try candidate starts from nominal_start - max_back up to nominal_start - 1 (nominal_start itself is not checked here)
        for (size_t cand_offset = max_back; cand_offset > 0; --cand_offset) {
            size_t cand = nominal_start - cand_offset; // candidate start bit position
            
            // Decode up to max_code_len_ bits starting from cand
            size_t try_end = std::min(cand + max_code_len_, total_bits);
            
            TrieNode* cur = trie;
            size_t p = cand;
            while (p < try_end && cur) {
                int b = fullBitstream[p] - '0';
                cur = cur->child[b];
                p++; // p is the bit position *after* the current bit
                
                if (cur && cur->symbol >= 0) {
                    // Found a symbol ending at bit position p
                    if (p > nominal_start) {
                        // The symbol decoded starts before nominal_start but ENDS AFTER nominal_start.
                        // This is a valid, aligned start for this block.
                        return cand;
                    }
                    // Symbol ended at or before nominal_start. Continue to the next symbol starting at p.
                    cur = trie;
                }
            }
        }
        // If no pre-boundary start was found, fall back to nominal_start.
        return nominal_start;
    }

    std::vector<uint8_t> decodeParallel(const std::vector<uint8_t>& compressed) {
        size_t index = 0;
        int padding = 0;
        std::unordered_map<uint8_t,std::string> codes;
        readHeader(compressed, index, padding, codes);
        
        // Need to set max_code_len_ from header if not set by encode
        if (max_code_len_ == 0) {
            for (auto &p : codes) if (p.second.size() > max_code_len_) max_code_len_ = p.second.size();
        }
        
        if (index > compressed.size()) return {};

        // build decode structures
        TrieNode* trie = buildTrie(codes);
        auto lut = buildLUT(codes);

        // convert payload to bitstring
        std::string fullBitstream;
        fullBitstream.reserve((compressed.size() - index) * 8);
        for (size_t i = index; i < compressed.size(); ++i) {
            uint8_t b = compressed[i];
            for (int bit=7; bit>=0; --bit) fullBitstream.push_back(((b>>bit)&1) ? '1' : '0');
        }
        if (padding > 0 && fullBitstream.size() >= static_cast<size_t>(padding))
            fullBitstream.resize(fullBitstream.size() - padding);

        size_t total_bits = fullBitstream.size();
        if (total_bits == 0) {
            freeTrie(trie);
            return {};
        }

        size_t bits_per_block = DECODE_BLOCK_SIZE * 8;
        size_t num_blocks = (total_bits + bits_per_block - 1) / bits_per_block;
        std::vector<std::vector<uint8_t>> decoded_blocks(num_blocks);

        auto t0 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic)
        for (size_t bidx = 0; bidx < num_blocks; ++bidx) {
            size_t start_bit_nominal = bidx * bits_per_block;
            size_t end_bit = std::min(start_bit_nominal + bits_per_block, total_bits);

            // align start bit: only apply alignment for blocks after the first one (bidx > 0)
            size_t aligned_start = start_bit_nominal;
            if (start_bit_nominal > 0) {
                aligned_start = align_start_bit(fullBitstream, start_bit_nominal, total_bits, trie, lut);
            }

            // decode from aligned_start up to end_bit
            std::vector<uint8_t> local_decoded = sequentialDecodeRange(fullBitstream, aligned_start, end_bit, trie, lut);

            // CRITICAL FIX: If alignment caused an overlap, skip the symbols that belong to the previous block.
            if (aligned_start < start_bit_nominal && !local_decoded.empty()) {
                // Re-simulate bit traversal from aligned_start to find the symbol index that corresponds to nominal_start
                size_t p = aligned_start;
                TrieNode* cur = trie;
                int symbols_to_skip = 0;
                bool keep_found = false;

                while (p < end_bit) {
                    
                    // Traverse one symbol bit-by-bit
                    size_t symbol_start_p = p;
                    TrieNode* current_node = trie;
                    size_t q = p;

                    while (q < end_bit && current_node) {
                        int b = fullBitstream[q] - '0';
                        current_node = current_node->child[b];
                        q++;
                        
                        if (current_node && current_node->symbol >= 0) {
                            // Symbol finished at bit position q
                            p = q; // Update main bit pointer to the end of the current symbol
                            
                            if (p > start_bit_nominal) {
                                // This is the first symbol that ENDS AFTER the nominal boundary. Keep it.
                                keep_found = true;
                                goto end_skip_loop;
                            } else {
                                // Symbol ended AT or BEFORE the nominal boundary. Skip it.
                                symbols_to_skip++;
                                break; // Break inner traversal, continue outer while loop
                            }
                        }
                    }
                    
                    if (!keep_found && q >= end_bit) {
                        break; // Reached end of block without finding a symbol to keep.
                    }
                }
                
                end_skip_loop:;

                // Apply the skip count to the pre-decoded vector
                if (keep_found && symbols_to_skip > 0 && symbols_to_skip < local_decoded.size()) {
                    // Erase the symbols that belong to the previous block.
                    local_decoded.erase(local_decoded.begin(), local_decoded.begin() + symbols_to_skip);
                    decoded_blocks[bidx] = std::move(local_decoded);
                } else if (symbols_to_skip == 0 && keep_found) {
                    // This can happen if the first symbol starts AT or near nominal_start and ends AFTER it. No skip needed.
                    decoded_blocks[bidx] = std::move(local_decoded);
                } else {
                    // All symbols were dropped, or block ended before finding a symbol to keep.
                    decoded_blocks[bidx].clear();
                }
            } else {
                // No alignment needed (bidx=0) or aligned_start == nominal_start. Keep all.
                decoded_blocks[bidx] = std::move(local_decoded);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        // assemble
        size_t total_out = 0;
        for (auto &v : decoded_blocks) total_out += v.size();
        std::vector<uint8_t> output;
        output.reserve(total_out);
        for (size_t i=0;i<decoded_blocks.size();++i) {
            output.insert(output.end(), decoded_blocks[i].begin(), decoded_blocks[i].end());
        }

        freeTrie(trie);

        auto micro = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        #pragma omp critical
        {
            std::cout << "Parallel decode elapsed: " << micro << " us (threads=" << omp_get_max_threads() << ")\n";
        }
        return output;
    }

    // ---------------- TREE & HEADER I/O ----------------

    Node* buildTree(const std::unordered_map<uint8_t,int>& freq) {
        std::priority_queue<Node*, std::vector<Node*>, NodeCompare> pq;
        for (auto &p : freq) pq.push(new Node(p.first, p.second));
        if (pq.empty()) return nullptr;
        // Handle single symbol case: ensure a two-node root for a valid code ("0")
        if (pq.size() == 1) { 
             Node* x = pq.top(); pq.pop(); 
             Node* dummy_leaf = new Node(0, 0); 
             return new Node(x, dummy_leaf); 
        }
        while (pq.size() > 1) {
            Node* a = pq.top(); pq.pop();
            Node* b = pq.top(); pq.pop();
            pq.push(new Node(a,b));
        }
        return pq.top();
    }

    void buildCodes(Node* n, const std::string& cur, std::unordered_map<uint8_t,std::string>& codes) {
        if (!n) return;
        if (n->isLeaf) { codes[n->symbol] = cur.empty() ? "0" : cur; return; }
        buildCodes(n->left,  cur + "0", codes);
        buildCodes(n->right, cur + "1", codes);
    }

    void freeTree(Node* n) {
        if (!n) return;
        freeTree(n->left); freeTree(n->right);
        delete n;
    }

    void writeHeader(std::vector<uint8_t>& out, int padding, const std::unordered_map<uint8_t,std::string>& codes) {
        out.push_back(static_cast<uint8_t>(padding));
        uint16_t N = static_cast<uint16_t>(codes.size());
        out.push_back((N>>8)&0xFF);
        out.push_back(N & 0xFF);
        for (auto &p : codes) {
            out.push_back(p.first);
            out.push_back(static_cast<uint8_t>(p.second.size()));
            for (char c : p.second) out.push_back(static_cast<uint8_t>(c));
        }
    }

    void readHeader(const std::vector<uint8_t>& in, size_t& index, int& padding, std::unordered_map<uint8_t,std::string>& codes) {
        if (index >= in.size()) throw std::runtime_error("Bad header");
        padding = in[index++];
        if (index + 1 >= in.size()) throw std::runtime_error("Bad header size");
        uint16_t N = (static_cast<uint16_t>(in[index])<<8) | static_cast<uint16_t>(in[index+1]);
        index += 2;
        for (uint16_t i=0;i<N;++i) {
            if (index + 1 >= in.size()) throw std::runtime_error("Bad header content");
            uint8_t sym = in[index++];
            uint8_t len = in[index++];
            std::string code; code.reserve(len);
            for (uint8_t j=0;j<len;++j) {
                if (index >= in.size()) throw std::runtime_error("Bad header code chars");
                code.push_back(static_cast<char>(in[index++]));
            }
            codes[sym] = code;
        }
    }

    int max_code_len_ = 0;
};

// ---------------- File IO utils ----------------
std::vector<uint8_t> read_file_to_vector(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("Cannot open: " + filename);
    std::streamsize size = in.tellg();
    if (size < 0) return {};
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    in.seekg(0,std::ios::beg);
    in.read(reinterpret_cast<char*>(buf.data()), size);
    return buf;
}
void write_vector_to_file(const std::string& filename, const std::vector<uint8_t>& data) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write: " + filename);
    out.write(reinterpret_cast<const char*>(data.data()), data.size());
}

int main(int argc, char** argv) {
    omp_set_num_threads(thread_count);
    HuffmanParallelCPU h;

    try {
        auto t_all_s = std::chrono::high_resolution_clock::now();
        auto data = read_file_to_vector("data.bin");
        auto t0 = std::chrono::high_resolution_clock::now();
        auto compressed = h.encode(data);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto decompressed = h.decode(compressed);
        auto t2 = std::chrono::high_resolution_clock::now();

        write_vector_to_file("compressed.huff", compressed);
        write_vector_to_file("decompressed.bin", decompressed);

        auto enc_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        auto dec_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count();

        bool ok = (data.size() == decompressed.size()) && std::equal(data.begin(), data.end(), decompressed.begin());

        std::cout << "\n=== Parallel CPU Results ===\n";
        std::cout << "Original size:  " << data.size() << " bytes\n";
        std::cout << "Compressed size:" << compressed.size() << " bytes\n";
        std::cout << "Compression ratio: " << (100.0 * compressed.size() / data.size()) << "%\n\n";
        std::cout << "Compression time:   " << enc_us << " us\n";
        std::cout << "Decompression time: " << dec_us << " us\n";
        std::cout << "Verification:       " << (ok ? "PASS" : "FAIL") << "\n";
        auto t_all_e = std::chrono::high_resolution_clock::now();
        auto wall_us = std::chrono::duration_cast<std::chrono::microseconds>(t_all_e - t_all_s).count();
        //std::cout << "Wall elapsed (main): " << wall_us << " us\n";
    } catch (const std::exception &ex) {
        std::cerr << "Exception: " << ex.what() << "\n";
        return 2;
    }
    return 0;
}