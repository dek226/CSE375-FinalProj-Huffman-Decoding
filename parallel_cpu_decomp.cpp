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

int thread_count = 16;
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
            while (p < end_bit && cur) {
                int b = fullBitstream[p] - '0';
                cur = cur->child[b];
                p++;
                if (cur && cur->symbol >= 0) {
                    out.push_back(static_cast<uint8_t>(cur->symbol));
                    pos = p;
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
        // try each candidate start from (nominal_start - max_back) up to nominal_start
        for (size_t back = 0; back <= max_back; ++back) {
            size_t cand = nominal_start - (max_back - back); // increasing candidate
            // try decode one symbol from cand; if the decoded symbol ends at or after nominal_start, accept
            // decode up to max_code_len_ bits
            size_t try_end = std::min(cand + max_code_len_, total_bits);
            std::vector<uint8_t> tmp = sequentialDecodeRange(fullBitstream, cand, try_end, trie, lut);
            if (!tmp.empty()) {
                // compute where the first symbol ended: decode length in bits = try_end - cand? Need exact
                // We can re-run bitwise to find first symbol end precisely
                TrieNode* cur = trie;
                size_t p = cand;
                while (p < try_end && cur) {
                    int b = fullBitstream[p] - '0';
                    cur = cur->child[b];
                    p++;
                    if (cur && cur->symbol >= 0) {
                        size_t symbol_end = p;
                        if (symbol_end >= nominal_start) return cand;
                        break; // first symbol finished earlier than nominal_start => not acceptable
                    }
                }
            }
        }
        // fallback: return nominal_start (will likely decode partial / stop quickly)
        return nominal_start;
    }

    std::vector<uint8_t> decodeParallel(const std::vector<uint8_t>& compressed) {
        size_t index = 0;
        int padding = 0;
        std::unordered_map<uint8_t,std::string> codes;
        readHeader(compressed, index, padding, codes);
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

            // align start bit by scanning within max_code_len_ bits (gap alignment)
            size_t aligned_start = start_bit_nominal;
            if (start_bit_nominal > 0) {
                aligned_start = align_start_bit(fullBitstream, start_bit_nominal, total_bits, trie, lut);
            }

            // decode from aligned_start up to end_bit (but avoid decoding symbols that end entirely before nominal start)
            auto local_decoded = sequentialDecodeRange(fullBitstream, aligned_start, end_bit, trie, lut);

            // If aligned_start < nominal start, drop decoded symbols that finish before nominal start
            if (aligned_start < start_bit_nominal && !local_decoded.empty()) {
                // To decide how many symbols to drop, re-simulate bit traversal until we reach nominal_start
                std::vector<uint8_t> kept;
                size_t p = aligned_start;
                TrieNode* cur = trie;
                while (p < end_bit) {
                    int b = fullBitstream[p] - '0';
                    cur = cur->child[b];
                    p++;
                    if (cur && cur->symbol >= 0) {
                        if (p > start_bit_nominal) {
                            // this symbol crosses the nominal start bit -> keep it and all subsequent
                            kept.push_back(static_cast<uint8_t>(cur->symbol));
                            // now append rest of decoded symbols by re-decoding the remainder
                            if (p < end_bit) {
                                auto remainder = sequentialDecodeRange(fullBitstream, p, end_bit, trie, lut);
                                kept.insert(kept.end(), remainder.begin(), remainder.end());
                            }
                            break;
                        } else {
                            // symbol ended before nominal start -> drop
                            cur = trie;
                            continue;
                        }
                    }
                }
                decoded_blocks[bidx] = std::move(kept);
            } else {
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
            // Note: we do not perform additional cross-block fixes here because alignment dropped overlapping symbols
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
        if (pq.size() == 1) { Node* x = pq.top(); pq.pop(); return new Node(x, new Node(0,0)); }
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
        std::cout << "Original size:  " << data.size() << " bytes\n";
        std::cout << "Compressed size:" << compressed.size() << " bytes\n";
        std::cout << "Compression ratio: " << (100.0 * compressed.size() / data.size()) << "%\n\n";
        std::cout << "Compression time:   " << enc_us << " us\n";
        std::cout << "Decompression time: " << dec_us << " us\n";
        std::cout << "Verification:       " << (ok ? "PASS" : "FAIL") << "\n";
        auto t_all_e = std::chrono::high_resolution_clock::now();
        auto wall_us = std::chrono::duration_cast<std::chrono::microseconds>(t_all_e - t_all_s).count();
        //std::cout << "Wall elapsed (main): " << wall_us << " us\n";
    } catch (const std::exception &ex) {
        std::cerr << "Exception: " << ex.what() << "\n";
        return 2;
    }
    return 0;
}
