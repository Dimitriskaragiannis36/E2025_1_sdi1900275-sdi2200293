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
public: //κονστράκτορας με τιμές ώστε να φτιάξει τα αντικείμενα
    IVFFlat(int kclusters = 4,
            int nprobe = 5,
            unsigned int seed = 1,
            int N = 1,
            float R = 2000.0f,
            utils::DistanceFunc dist_func = nullptr);

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
    int nlist() const { return  kclusters_; }
    int nprobe() const { return nprobe_; }
    const std::vector<std::vector<float>>& centroids() const { return centroids_; }
    const std::vector<std::vector<float>>& data() const { return *data_ptr_; }
    utils::DistanceFunc distance_func() const { return dist_func_; }

private:
    int kclusters_;    //αριθμός clusters
    int nprobe_;       //αριθμός clusters που εξετάζονται
    unsigned int seed_;
    int N_;            //default N για KNN
    float R_;          //default R για range search
    utils::DistanceFunc dist_func_;

    std::vector<std::vector<float>> centroids_;     //C = {c1, ..., ck}
    std::vector<std::vector<int>> inverted_lists_;  //IL_j = ids των points
    const std::vector<std::vector<float>>* data_ptr_; //pointer στα data

};

} //namespace ivf

#endif //IVFF_H
