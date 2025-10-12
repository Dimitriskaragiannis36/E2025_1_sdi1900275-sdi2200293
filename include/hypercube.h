#ifndef HYPERCUBE_H
#define HYPERCUBE_H

#include "helper.h"
#include <vector>
#include <utility>

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

    /*k-NN(top-N)*/
    std::vector<std::pair<int, float>> knn_query(const std::vector<float> &q, int N) const;

    /*range(ακτίνα R)*/
    std::vector<int> range_search(const std::vector<float> &q, float R) const;

    /*απελευθέρωση πόρων (no-op)*/
    void clear_index();
  };

} /*namespace nn*/

#endif // HYPERCUBE_H