#include "helper.h"
#include "helper.tpp"
#include <iostream>
#include <cstdlib>

namespace utils
{
    /*συνάρτηση για την ανάλυση των παραμέτρων από τη γραμμή εντολών*/
    Params parse_args(int argc, char *argv[])
    {
        Params p;
        for (int i = 1; i < argc; i++)
        {
            std::string arg = argv[i];

            if (arg == "-d" && i + 1 < argc)
                p.input_file = argv[++i];
            else if (arg == "-q" && i + 1 < argc)
                p.query_file = argv[++i];
            else if (arg == "-o" && i + 1 < argc)
                p.output_file = argv[++i];
            else if (arg == "-k" && i + 1 < argc)
                p.k = atoi(argv[++i]);
            else if (arg == "-L" && i + 1 < argc)
                p.L = atoi(argv[++i]);
            else if (arg == "-w" && i + 1 < argc)
                p.w = atoi(argv[++i]);
            else if (arg == "-N" && i + 1 < argc)
                p.N = atoi(argv[++i]);
            else if (arg == "-R" && i + 1 < argc)
                p.R = atof(argv[++i]);
            else if (arg == "-type" && i + 1 < argc)
                p.type = argv[++i];
            else if (arg == "-range" && i + 1 < argc)
                p.do_range = (std::string(argv[++i]) == "true");
            else if (arg == "-seed" && i + 1 < argc)
                p.seed = atoi(argv[++i]);

            /*hypercube και κοινά switches*/
            else if (arg == "-hypercube")
                p.use_hypercube = true;
            else if (arg == "-kproj" && i + 1 < argc)
                p.kproj = std::stoi(argv[++i]);
            else if (arg == "-M" && i + 1 < argc)
            {
                /*Σημείωση: Το -M είναι είτε το max_candidates του Hypercube είτε το M του IVFPQ.
                    Το αποθηκεύουμε και στα δύο. Το ενεργό εξαρτάται από τη λειτουργία.*/
                int val = std::stoi(argv[++i]);
                p.max_candidates = val;
                p.pq_M = val;
            }
            else if (arg == "-probes" && i + 1 < argc)
                p.max_probes = std::stoi(argv[++i]);

            /*νέα flags για IVFFlat / IVFPQ*/
            else if (arg == "-ivfflat")
                p.use_ivfflat = true;
            else if (arg == "-ivfpq")
                p.use_ivfpq = true;
            else if (arg == "-kclusters" && i + 1 < argc)
                p.kclusters = atoi(argv[++i]);
            else if (arg == "-nprobe" && i + 1 < argc)
                p.nprobe = atoi(argv[++i]);
            else if (arg == "-nbits" && i + 1 < argc)
                p.pq_nbits = atoi(argv[++i]);
            /*LSH*/
            else if (arg == "-lsh")
                p.use_lsh = true;
            /*άγνωστη παράμετρος*/
            else
            {
                std::cerr << "Warning: Unknown argument '" << arg << "' ignored.\n";
            }
        }

        /*προεπιλεγμένο R ανάλογα με το σύνολο δεδομένων*/
        if (p.type == "sift")
            p.R = 2.0f;
        else
            p.R = 2000.0f;

        return p;
    }

    /*συνάρτηση για την εκτύπωση των παραμέτρων*/
    void print_params(const Params &p)
    {
        std::cout << "\n=========================\n";
        std::cout << "      RUN PARAMETERS     \n";
        std::cout << "=========================\n";

        std::cout << "Input file        : " << (p.input_file.empty() ? "(none)" : p.input_file) << "\n";
        std::cout << "Query file        : " << (p.query_file.empty() ? "(none)" : p.query_file) << "\n";
        std::cout << "Output file       : " << (p.output_file.empty() ? "(none)" : p.output_file) << "\n";
        std::cout << "Dataset type      : " << (p.type.empty() ? "(none)" : p.type) << "\n\n";

        std::cout << "General:\n";
        std::cout << "  seed            = " << p.seed << "\n";
        std::cout << "  N (neighbors)   = " << p.N << "\n";
        std::cout << "  R (radius)      = " << p.R << "\n";
        std::cout << "  range search    = " << (p.do_range ? "true" : "false") << "\n\n";

        std::cout << "Modes enabled:\n";
        std::cout << "  use_lsh         = " << (p.use_lsh ? "true" : "false") << "\n";
        std::cout << "  use_hypercube   = " << (p.use_hypercube ? "true" : "false") << "\n";
        std::cout << "  use_ivfflat     = " << (p.use_ivfflat ? "true" : "false") << "\n";
        std::cout << "  use_ivfpq       = " << (p.use_ivfpq ? "true" : "false") << "\n\n";

        if (p.use_lsh)
        {
            std::cout << "[LSH parameters]\n";
            std::cout << "  k (hash funcs)  = " << p.k << "\n";
            std::cout << "  L (tables)      = " << p.L << "\n";
            std::cout << "  w (window)      = " << p.w << "\n\n";
        }

        if (p.use_hypercube)
        {
            std::cout << "[Hypercube parameters]\n";
            std::cout << "  kproj           = " << p.kproj << "\n";
            std::cout << "  w               = " << p.w << "\n";
            std::cout << "  M (max cand.)   = " << p.max_candidates << "\n";
            std::cout << "  probes          = " << p.max_probes << "\n\n";
        }

        if (p.use_ivfflat)
        {
            std::cout << "[IVFFlat parameters]\n";
            std::cout << "  kclusters       = " << p.kclusters << "\n";
            std::cout << "  nprobe          = " << p.nprobe << "\n\n";
        }

        if (p.use_ivfpq)
        {
            std::cout << "[IVFPQ parameters]\n";
            std::cout << "  kclusters       = " << p.kclusters << "\n";
            std::cout << "  nprobe          = " << p.nprobe << "\n";
            std::cout << "  M (subvectors)  = " << p.pq_M << "\n";
            std::cout << "  nbits           = " << p.pq_nbits << "\n\n";
        }

        std::cout << "=========================\n\n";
    }

    /*συνάρτηση για τον υπολογισμό της ευκλείδειας απόστασης μεταξύ δύο σημείων*/
    float euclidean_distance(const std::vector<float> &x, const std::vector<float> &y)
    {
        float dist = 0.0f;
        for (size_t i = 0; i < x.size(); ++i)
        {
            float diff = x[i] - y[i];
            dist += diff * diff;
        }
        return std::sqrt(dist);
    }

} /*namespace utils*/