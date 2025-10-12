#ifndef IVFF_H
#define IVFF_H

#include "helper.h"
#include "kmeans.h"
#include "silhouette.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <limits>
#include <utility>

namespace ivf {

//IVF-Flat Index
class IVFFlat {
public:
    IVFFlat(int nlist, int nprobe, unsigned int seed = 1,
            utils::DistanceFunc dist_func = utils::euclidean_distance);

    //κατασκευή του index
    void build_index(const std::vector<std::vector<float>>& data);

    //αναζήτηση Ν πλησιέστερων γειτόνων (για run_queries)
    std::vector<std::pair<int,float>> knn_query(const std::vector<float>& q, int N) const;

    //range Search (για run_queries)
    std::vector<int> range_search(const std::vector<float>& q, float R) const;

    //επιστροφή υποψηφίων (για generic utils::knn_query / range_search)
    std::vector<int> query_candidates(const std::vector<float>& q) const;

    //καθαρισμός
    void clear_index();

    //getters
    int nlist() const { return nlist_; }
    int nprobe() const { return nprobe_; }
    const std::vector<std::vector<float>>& centroids() const { return centroids_; }
    const std::vector<std::vector<float>>& data() const { return *data_ptr_; }
    utils::DistanceFunc distance_func() const { return dist_func_; }

private:
    int nlist_;    //αριθμός clusters (inverted lists)
    int nprobe_;   //αριθμός clusters που εξετάζονται ανά query
    unsigned int seed_;
    utils::DistanceFunc dist_func_;

    std::vector<std::vector<float>> centroids_;     //C = {c1, ..., ck}
    std::vector<std::vector<int>> inverted_lists_;  //IL_j = ids των points
    const std::vector<std::vector<float>>* data_ptr_; //pointer στα data

};

} //namespace ivf

#endif //IVFF_H
