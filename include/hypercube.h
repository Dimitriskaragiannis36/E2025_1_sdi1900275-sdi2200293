#ifndef HYPERCUBE_H
#define HYPERCUBE_H

#include "helper.h"
#include <vector>
#include <utility>
#include <random>

namespace nn
{

  /*Α2*/
  /*προσθήκη συνάρτησης κατακερματισμού h_i(p) = floor((v_i·p + t_i)/w).*/
  class Hypercube
  {
  public:
    Hypercube(int dim,
              int kproj = 14,
              int w = 4,
              int M = 10,
              int probes = 2,
              unsigned int seed = 1,
              utils::DistanceFunc dist_func = utils::euclidean_distance);

    /*δημιουργία δείκτη πάνω στο dataset*/
    void build_index(const std::vector<std::vector<float>> &data);

    /*k-NN (επιστρέφει ζεύγη (index, απόσταση))*/
    std::vector<std::pair<int, float>> knn_query(const std::vector<float> &q, int N) const;

    /*ακτίνα (επιστρέφει indices)*/
    std::vector<int> range_search(const std::vector<float> &q, float R) const;

    /*καθαρισμός πόρων*/
    void clear_index();

  private:
    int dim_;   /*διαστατικότητα πρωτογενούς χώρου*/
    int kproj_; /*αριθμός προβολών (k')*/
    int w_;     /*παράμετρος πλάτους κουβάδων*/
    int M_;
    int probes_;
    unsigned int seed_;             /*σπόρος RNG*/
    utils::DistanceFunc dist_func_; /*μετρική απόστασης*/

    /*τυχαιοποιητής & κατανομές*/
    std::mt19937 rng_;
    std::normal_distribution<float> normal_;    /*για v ~ N(0,1)*/
    std::uniform_real_distribution<float> uni_; /*για t ~ U(0,w)*/

    /*προβολές και μετατοπίσεις*/
    std::vector<std::vector<float>> v_; /*v_[i].size()==dim_*/
    std::vector<float> t_;              /* t_[i] ∈ [0, w) */

    /*δείκτης στο dataset*/
    const std::vector<std::vector<float>> *data_ptr_ = nullptr;

    /*υπολογισμός των k' ακέραιων τιμών h_i(p) για ένα διάνυσμα p*/
    std::vector<long long> compute_hashes(const std::vector<float> &p) const;
  };

} /*namespace nn*/

#endif /*HYPERCUBE_H*/
