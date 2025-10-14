#include "helper.h"
#include "helper.tpp"
#include <cstdlib>

namespace utils
{

    /*parse_args: Ανάλυση παραμέτρων CLI
     * προσθήκη των flags του Hypercube (A2):
     * -hypercube -kproj -M -probes (το -w είναι κοινό με LSH)*/

    Params parse_args(int argc, char *argv[])
    {
        Params p;

        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            /*αρχεία / τύπος δεδομένων*/
            if (arg == "-d" && i + 1 < argc)
                p.input_file = argv[++i];
            else if (arg == "-q" && i + 1 < argc)
                p.query_file = argv[++i];
            else if (arg == "-o" && i + 1 < argc)
                p.output_file = argv[++i];
            else if (arg == "-type" && i + 1 < argc)
            {
                p.data_type = argv[++i];
                /*προσαρμογή default R βάσει τύπου*/
                if (p.data_type == "sift")
                    p.R = 2.0f;
                else if (p.data_type == "mnist")
                    p.R = 2000.0f;
            }

            else if (arg == "-N" && i + 1 < argc)
                p.N = std::atoi(argv[++i]);
            else if (arg == "-R" && i + 1 < argc)
                p.R = static_cast<float>(std::atof(argv[++i]));
            else if (arg == "-range" && i + 1 < argc)
            {
                std::string v = argv[++i];
                for (auto &c : v)
                    c = std::tolower(c);
                p.do_range = (v == "true" || v == "1" || v == "yes");
            }
            else if (arg == "-seed" && i + 1 < argc)
                p.seed = static_cast<unsigned int>(std::atoi(argv[++i]));

            /*LSH (A1)*/
            else if (arg == "-lsh")
                p.use_lsh = true;
            else if (arg == "-k" && i + 1 < argc)
                p.k = std::atoi(argv[++i]);
            else if (arg == "-L" && i + 1 < argc)
                p.L = std::atoi(argv[++i]);
            else if (arg == "-w" && i + 1 < argc)
                p.w = std::atoi(argv[++i]);

            /*Hypercube (A2)*/
            else if (arg == "-hypercube")
                p.use_hypercube = true;
            else if (arg == "-kproj" && i + 1 < argc)
                p.kproj = std::atoi(argv[++i]);
            else if (arg == "-M" && i + 1 < argc)
                p.M_cap = std::atoi(argv[++i]);
            else if (arg == "-probes" && i + 1 < argc)
                p.probes = std::atoi(argv[++i]);

            /*IVFFlat (A3)*/
            else if (arg == "-ivfflat")
                p.use_ivfflat = true;
            else if (arg == "-kclusters" && i + 1 < argc)
                p.kclusters = std::atoi(argv[++i]);
            else if (arg == "-nprobe" && i + 1 < argc)
                p.nprobe = std::atoi(argv[++i]);

            /*αγνόηση άγνωστων flags*/
        }

        return p;
    }

    /*print_params*/
    void print_params(const Params &p)
    {
        std::cout << "Input   : " << p.input_file << "\n";
        std::cout << "Queries : " << p.query_file << "\n";
        std::cout << "Output  : " << p.output_file << "\n";
        std::cout << "Type    : " << p.data_type << "\n";
        std::cout << "N       : " << p.N << "\n";
        std::cout << "R       : " << p.R << "\n";
        std::cout << "range   : " << (p.do_range ? "true" : "false") << "\n";
        std::cout << "seed    : " << p.seed << "\n";

        if (p.use_lsh)
        {
            std::cout << " [LSH] k=" << p.k
                      << " L=" << p.L
                      << " w=" << p.w << "\n";
        }
        if (p.use_hypercube)
        {
            std::cout << " [Hypercube] kproj=" << p.kproj
                      << " w=" << p.w
                      << " M=" << p.M_cap
                      << " probes=" << p.probes << "\n";
        }
        if (p.use_ivfflat)
        {
            std::cout << " [IVFFlat] kclusters=" << p.kclusters
                      << " nprobe=" << p.nprobe << "\n";
        }
    }

    /* Ευκλείδεια απόσταση (L2) */
    float euclidean_distance(const std::vector<float> &x,
                             const std::vector<float> &y)
    {
        float dist = 0.0f;
        const std::size_t n = x.size();
        for (std::size_t i = 0; i < n; ++i)
        {
            const float diff = x[i] - y[i];
            dist += diff * diff;
        }
        return std::sqrt(dist);
    }

} /*namespace utils*/
