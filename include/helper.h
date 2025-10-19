#ifndef HELPER_H
#define HELPER_H

#include <vector>
#include <string>
#include <fstream>
#include <functional>
#include <cmath>
#include <unordered_set>

namespace utils
{
    /*τύπος για συνάρτηση απόστασης*/
    using DistanceFunc = std::function<float(const std::vector<float> &, const std::vector<float> &)>;

    /*παράμετροι από γραμμή εντολών (και default τιμές) για ασφάλεια*/
    struct Params
    {
        /*κοινές παράμετροι*/
        int seed = 1;
        int N = 1;
        float R = 2000.0f;
        std::string input_file = "data/input.dat";
        std::string query_file = "data/query.dat";
        std::string output_file = "results.txt";
        std::string type = "mnist";
        bool do_range = true;

        /*LSH*/
        int k = 4;
        int L = 5;
        int w = 4;
        bool use_lsh = false;

        /*Hypercube*/
        bool use_hypercube = false;
        int kproj = 14;
        int max_candidates = 10;
        int max_probes = 2;

        /*IVFFlat*/
        bool use_ivfflat = false;
        int kclusters = 50;
        int nprobe = 5;

        /*IVFPQ*/
        bool use_ivfpq = false;
        int pq_M = 16;    /*αριθμός υποδιανυσμάτων M*/
        int pq_nbits = 8; /*μέγεθος codebook = 2^nbits ανά υπο-διανύσμα*/
    };

    Params parse_args(int argc, char *argv[]);
    void print_params(const Params &p);

    template <typename IndexType>
    void run_queries(IndexType &index,
                     const std::vector<std::vector<float>> &data,
                     const std::vector<std::vector<float>> &queries,
                     const Params &params,
                     std::ofstream &out);

    float euclidean_distance(const std::vector<float> &x, const std::vector<float> &y);

} /*namespace utils*/

#include "helper.tpp"

#endif /*HELPER_H*/