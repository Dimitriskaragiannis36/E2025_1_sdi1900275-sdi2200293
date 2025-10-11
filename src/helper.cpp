#include "helper.h"
#include "helper.tpp"
#include <iostream>
#include <cstdlib>

namespace utils {

//συνάρτηση για την ανάλυση των παραμέτρων από τη γραμμή εντολών
Params parse_args(int argc, char* argv[]) {
    Params p;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-d" && i+1 < argc) p.input_file = argv[++i];
        else if (arg == "-q" && i+1 < argc) p.query_file = argv[++i];
        else if (arg == "-o" && i+1 < argc) p.output_file = argv[++i];
        else if (arg == "-k" && i+1 < argc) p.k = atoi(argv[++i]);
        else if (arg == "-L" && i+1 < argc) p.L = atoi(argv[++i]);
        else if (arg == "-w" && i+1 < argc) p.w = atoi(argv[++i]);
        else if (arg == "-N" && i+1 < argc) p.N = atoi(argv[++i]);
        else if (arg == "-R" && i+1 < argc) p.R = atof(argv[++i]);
        else if (arg == "-type" && i+1 < argc) p.type = argv[++i];
        else if (arg == "-lsh") p.use_lsh = true;
        else if (arg == "-range" && i+1 < argc) p.do_range = (std::string(argv[++i]) == "true");
        else if (arg == "-seed" && i+1 < argc) p.seed = atoi(argv[++i]);
    }
    if (p.type == "sift") p.R = 2.0f;
    return p;
}

//συνάρτηση για την εκτύπωση των παραμέτρων
void print_params(const Params& p) {
    std::cout << "Parameters: "
        << "seed=" << p.seed << " "
        << "k=" << p.k << " "
        << "L=" << p.L << " "
        << "w=" << p.w << " "
        << "N=" << p.N << " "
        << "R=" << p.R << " "
        << "type=" << p.type << " "
        << "input=" << p.input_file << " "
        << "query=" << p.query_file << " "
        << "output=" << p.output_file << " "
        << "range=" << (p.do_range ? "true" : "false") << std::endl;
}

//συνάρτηση για τον υπολογισμό της ευκλείδειας απόστασης μεταξύ δύο σημείων
float euclidean_distance(const std::vector<float>& x, const std::vector<float>& y) {
    float dist = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        float diff = x[i] - y[i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

} // namespace utils
