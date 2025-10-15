#ifndef HYPERCUBE_H
#define HYPERCUBE_H

#include "helper.h"
#include <unordered_map>
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>

namespace hcube
{

  /*Hypercube*/
  /*Περιγραφή:
      Υλοποιεί τον αλγόριθμο Approximate Nearest Neighbor
      χρησιμοποιώντας δομή Random Projection Hypercube.
      Κάθε σημείο προβάλλεται σε k' (dprime) τυχαίες διαστάσεις
      και αντιστοιχίζεται σε μία κορυφή του δυαδικού κύβου (vertex).*/
  class Hypercube
  {
  public:
    Hypercube(int dim, int dprime, int w, int max_candidates, int max_probes,
              unsigned int seed, utils::DistanceFunc dist_func);

    /*Δημιουργία του δείκτη (ευρετηρίου):
      - Υπολογίζει για κάθε σημείο το vertex id
      - Αποθηκεύει το index του σημείου στον κατάλληλο κάδο (vertex_buckets_)*/
    void build_index(const std::vector<std::vector<float>> &data);

    /*διαγραφή όλων των κάδων/προβολών*/
    void clear_index();

    /*k-Nearest Neighbors:επιστρέφει N κοντινότερους γείτονες (index, απόσταση)*/
    std::vector<std::pair<int, float>> knn_query(const std::vector<float> &q, int N) const;

    /*range search:επιστρέφει όλα τα σημεία εντός ακτίνας R*/
    std::vector<int> range_search(const std::vector<float> &q, float R) const;

    /*πιστρέφει όλους τους υποψηφίους γείτονες που ανήκουν στις κοντινές κορυφές (ανάλογα με το Hamming distance)*/
    std::vector<int> query_candidates(const std::vector<float> &q) const;

    /*πρόσβαση στα δεδομένα και στη συνάρτηση απόστασης*/
    const std::vector<std::vector<float>> &data() const { return *data_ptr_; }
    utils::DistanceFunc distance_func() const { return dist_func_; }

  private:
    int dim_;                       /*αρχική διαστατικότητα*/
    int dprime_;                    /*αριθμός προβολών (bits του hypercube)*/
    int w_;                         /*πλάτος bucket (hash width)*/
    int max_candidates_;            /*μέγιστος αριθμός υποψηφίων σημείων*/
    int max_probes_;                /*μέγιστος αριθμός κορυφών για probing*/
    unsigned int seed_;             /*σπόρος RNG*/
    utils::DistanceFunc dist_func_; /*συνάρτηση απόστασης (π.χ. L2)*/

    /*δεδομένα του dataset*/
    const std::vector<std::vector<float>> *data_ptr_ = nullptr;

    /*Τυχαίες προβολές τύπου LSH:
      - v_ -> k' διανύσματα προβολής (Gaussian)
      - t_ -> μετατοπίσεις (Uniform[0, w))
      - bit_map_ -> αντιστοίχιση ακέραιου hash -> bit {0,1}*/
    std::vector<std::vector<float>> v_;
    std::vector<float> t_;
    mutable std::vector<std::unordered_map<long long, uint8_t>> bit_map_;

    /*vertex_buckets_: κάθε vertex (64-bit id) δείχνει σε λίστα σημείων (indices)*/
    std::unordered_map<uint64_t, std::vector<int>> vertex_buckets_;

    /*Εσωτερικές βοηθητικές συναρτήσεις:
        - point_to_vertex(p) -> υπολογίζει σε ποια κορυφή ανήκει το p
        - collect_neighbors_by_hamming(v, out) -> συλλέγει υποψηφίους από κοντινές κορυφές (με αύξουσα απόσταση Hamming)*/
    uint64_t point_to_vertex(const std::vector<float> &p) const;
    void collect_neighbors_by_hamming(uint64_t vertex, std::vector<int> &out_candidates) const;
  };

} /*namespace hcube*/

#endif /*HYPERCUBE_H*/
