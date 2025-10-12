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
        else if (arg == "-range" && i+1 < argc) p.do_range = (std::string(argv[++i]) == "true");
        else if (arg == "-seed" && i+1 < argc) p.seed = atoi(argv[++i]);

        //νέα flags για IVFFlat / IVFPQ
        else if (arg == "-ivfflat") p.use_ivfflat = true;
        //else if (arg == "-ivfpq") p.use_ivfpq = true;

        else if (arg == "-kclusters" && i+1 < argc) p.kclusters = atoi(argv[++i]);
        else if (arg == "-nprobe" && i+1 < argc) p.nprobe = atoi(argv[++i]);

        //υπάρχουσα επιλογή για LSH
        else if (arg == "-lsh") p.use_lsh = true;
        //άγνωστη παράμετρος
        else {
            std::cerr << "Warning: Unknown argument '" << arg << "' ignored.\n";
        }

    }

    //default R ανάλογα με το dataset
    if (p.type == "sift") p.R = 2.0f;
    else p.R = 2000.0f;

    return p;
}


//συνάρτηση για την εκτύπωση των παραμέτρων
void print_params(const Params& p) {
    std::cout << "Parameters:\n";
    std::cout << " seed=" << p.seed
              << " N=" << p.N
              << " R=" << p.R
              << " type=" << p.type << "\n";
    std::cout << " input=" << p.input_file
              << " query=" << p.query_file
              << " output=" << p.output_file << "\n";
    if (p.use_lsh) {
        std::cout << " [LSH mode] k=" << p.k
                  << " L=" << p.L
                  << " w=" << p.w << "\n";
    }
    if (p.use_ivfflat) {
        std::cout << " [IVFFlat mode] kclusters=" << p.kclusters
                  << " nprobe=" << p.nprobe << "\n";
    }
    /*if (p.use_ivfpq) {
        std::cout << " [IVFPQ mode] kclusters=" << p.kclusters
                  << " nprobe=" << p.nprobe << "\n";
    }*/
    std::cout << " range=" << (p.do_range ? "true" : "false") << "\n";
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
