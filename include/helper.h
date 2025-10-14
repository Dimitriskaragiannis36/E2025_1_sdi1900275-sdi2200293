#ifndef HELPER_H
#define HELPER_H

#include <vector>
#include <string>
#include <functional>
#include <unordered_set>
#include <iostream>
#include <cmath>

namespace utils
{

    /*Συνάρτηση απόστασης (τύπος)*/
    using DistanceFunc = std::function<float(const std::vector<float> &,
                                             const std::vector<float> &)>;

    /*Δομή παραμέτρων προγράμματος (CLI)*/
    /*προσθήκη τα flags του Hypercube*/
    /*-hypercube -kproj -M -probes (-w επαναχρησιμοποιείται)*/
    struct Params
    {
        /*γενικά αρχεία / επιλογές*/
        std::string input_file;
        std::string query_file;
        std::string output_file;
        std::string data_type = "mnist";

        int N = 1;             /*-N: #γειτόνων για k-NN*/
        float R = 2000.0f;     /*-R: ακτίνα για range search*/
        bool do_range = true;  /*-range true|false*/
        unsigned int seed = 1; /*-seed*/

        /*LSH (A1)*/
        bool use_lsh = false; /*-lsh*/
        int k = 4;            /*-k  (LSH concatenations)*/
        int L = 5;            /*-L  (LSH tables)*/
        int w = 4;            /*-w  (LSH / Hypercube πλάτος bucket)*/

        /*Hypercube (A2)*/
        bool use_hypercube = false; /*-hypercube*/
        int kproj = 14;             /*-kproj: #τυχαίες προβολές (k')*/
        int M_cap = 10;             /*-M: όριο υποψηφίων*/
        int probes = 2;             /*-probes: #κορυφών για probing*/

        /*IVFFlat (A3)*/
        bool use_ivfflat = false; /*-ivfflat*/
        int kclusters = 32;       /*-kclusters*/
        int nprobe = 1;           /*-nprobe*/
    };

    /*δηλώσεις συναρτήσεων βοηθητικών*/
    Params parse_args(int argc, char *argv[]);
    void print_params(const Params &p);

    /*ευκλείδεια απόσταση*/
    float euclidean_distance(const std::vector<float> &x,
                             const std::vector<float> &y);

} /*namespace utils*/

#endif /*HELPER_H*/
