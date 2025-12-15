#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <iostream>


//generate data of any size and redundancy
int main(int argc, char** argv){
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <filesize> <redundancy>\n"
                  << "  filesize   : number of bytes (e.g., 100000000 for 100MB)\n"
                  << "  redundancy : 0.0 to 1.0\n";
        return 1;
    }
    size_t filesize = std::stoull(argv[1]);
    double redundancy = std::stod(argv[2]);

    redundancy = std::max(0.0, std::min(1.0, redundancy));

    std::string filename = "data.bin";

    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Error: cannot open output file: " + filename);
    }

    // RNG setup
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist_full(0, 255);
    std::uniform_int_distribution<int> dist_small(0, 3); // A–D

    for (size_t i = 0; i < filesize; ++i) {
        double r = std::generate_canonical<double, 10>(rng);

        uint8_t value;

        if (r < redundancy) {
            // Low-entropy A–D
            value = static_cast<uint8_t>('A' + dist_small(rng));
        } else {
            // High-entropy random byte
            value = static_cast<uint8_t>(dist_full(rng));
        }

        out.put(static_cast<char>(value));  
        if (!out) {
            throw std::runtime_error("Error writing to file: " + filename);
        }
    }

    out.close();
    std::cout << "Generated " << filesize << " bytes in " << filename << "\n";
    return 0;
}
