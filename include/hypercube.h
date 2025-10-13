#ifndef HYPERCUBE_H
#define HYPERCUBE_H

#include "helper.h"
#include <vector>
#include <utility>
#include <random>

namespace nn
{
  /*A2*/
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

    /*κατασκευή ευρετηρίου πάνω στο dataset*/
    void build_index(const std::vector<std::vector<float>> &data);

    /*k-NN (top-N)*/
    std::vector<std::pair<int, float>> knn_query(const std::vector<float> &q, int N) const;

    /*range (ακτίνα R)*/
    std::vector<int> range_search(const std::vector<float> &q, float R) const;

    /*απελευθέρωση πόρων*/
    void clear_index();

  private:
    /*RNG και προβολές*/
    int dim_;
    int kproj_;
    int w_;
    int M_;
    int probes_;
    unsigned int seed_;
    utils::DistanceFunc dist_func_;

    /*RNG & κατανομές*/
    std::mt19937 rng_;
    std::normal_distribution<float> normal_;
    std::uniform_real_distribution<float> uni_;

    /*τυχαίες προβολές και μετατοπίσεις*/
    /*v_[i] έχει μήκος dim_, κληρώνεται ~ N(0,1)*/
    std::vector<std::vector<float>> v_;
    /*t_[i] κληρώνεται ~ Uniform(0, w)*/
    std::vector<float> t_;

    /*δείκτης στο dataset (ορίζεται στο build_index;)*/
    const std::vector<std::vector<float>> *data_ptr_ = nullptr;
  };

} /*namespace nn*/

#endif /*HYPERCUBE_H*/
