#ifndef HELPER_H 
#define HELPER_H

#include <vector> //για std::vector
#include <string> //για std::string
#include <fstream> //για std::ofstream
#include <functional> //για std::function
#include <cmath> //για std::sqrt

namespace utils
{
    /*τύπος για συνάρτηση απόστασης*/
    using DistanceFunc = std::function<float(const std::vector<float> &, const std::vector<float> &)>;

    //σταθερά για το αν κανονικοποιούμε MNIST ----
    constexpr bool MNIST_NORMALIZED = false;
    
    /*παράμετροι από γραμμή εντολών (και default τιμές) για ασφάλεια*/
    struct Params
    {
        /*κοινές παράμετροι*/
        int seed = 1;  //αρχικοποίηση γεννήτριας τυχαίων αριθμών
        int N = 1;   //αριθμός γειτόνων
        float R = 2000.0f;
        std::string input_file = "data/input.dat"; //προεπιλεγμένο σύνολο δεδομένων
        std::string query_file = "data/query.dat"; //προεπιλεγμένο σύνολο ερωτημάτων
        std::string output_file = "results.txt";  //προεπιλεγμένο αρχείο εξόδου
        std::string type = "mnist";  //τύπος δεδομένων: "mnist" ή "sift"
        bool do_range = true;  //αν θα εκτελεστεί range search

        /*LSH*/
        int k = 4; //αριθμός συναρτήσεων κατακερματισμού ανά πίνακα
        int L = 5; //αριθμός πινάκων κατακερματισμού
        int w = 4; //παράμετρος παραθύρου
        bool use_lsh = false; //αν θα χρησιμοποιηθεί LSH

        /*Hypercube*/
        bool use_hypercube = false; //αν θα χρησιμοποιηθεί Hypercube
        int kproj = 14; //αριθμός διαστάσεων προβολής
        int max_candidates = 10; //μέγιστος αριθμός υποψηφίων
        int max_probes = 2; //μέγιστος αριθμός probes

        /*IVFFlat*/
        bool use_ivfflat = false; //αν θα χρησιμοποιηθεί IVFFlat
        int kclusters = 50; //αριθμός κέντρων (clusters)
        int nprobe = 5; //αριθμός probes

        /*IVFPQ*/
        bool use_ivfpq = false; //αν θα χρησιμοποιηθεί IVFPQ
        int pq_M = 16;    /*αριθμός υποδιανυσμάτων M*/
        int pq_nbits = 8; /*μέγεθος codebook = 2^nbits ανά υπο-διανύσμα*/
    };

    //ανάλυση παραμέτρων από γραμμή εντολών
    Params parse_args(int argc, char *argv[]); 

    //εκτύπωση παραμέτρων εκτέλεσης
    void print_params(const Params &p); 

    //συνάρτηση για εκτέλεση ερωτημάτων
    template <typename IndexType> 
    void run_queries(IndexType &index,
                     const std::vector<std::vector<float>> &data,
                     const std::vector<std::vector<float>> &queries,
                     const Params &params,
                     std::ofstream &out);

    //υπολογισμός Ευκλείδειας απόστασης
    float euclidean_distance(const std::vector<float> &x, const std::vector<float> &y);

} /*namespace utils*/

#include "helper.tpp" //συμπερίληψη υλοποίησης συναρτήσεων template

#endif /*HELPER_H*/