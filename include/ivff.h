#ifndef IVFF_H
#define IVFF_H

#include <vector>
#include <cstdint>

namespace nn {

class IVFFlat {
public:
    //κονστράκτορας
    IVFFlat(int dim, int nlist = 100, int nprobe = 1, int max_kmeans_iter = 20, unsigned int seed = 12345);

    //κατασκευή ευρετηρίου
    void build_index(const std::vector<std::vector<float>>& data);

    //εύρεση k-πλησιέστερων γειτόνων
    std::vector<std::pair<int,float>> knn_query(const std::vector<float>& q, int N) const;

    //εύρεση εντός ακτίνας
    std::vector<int> range_search(const std::vector<float>& q, float R) const;

    //καθαρισμός ευρετηρίου
    void clear_index();

private:
    int dim_;
    int nlist_;
    int nprobe_;
    int max_kmeans_iter_;
    unsigned int seed_;

    //κεντροειδή
    std::vector<std::vector<float>> centroids_;

    //λίστες δεικτών
    std::vector<std::vector<int>> lists_;

    //δεδομένα
    const std::vector<std::vector<float>>* data_ptr_ = nullptr;

    //βοηθητικές συναρτήσεις
    float euclidean_distance_sq(const std::vector<float>& a, const std::vector<float>& b) const;
    
    void run_kmeans(const std::vector<std::vector<float>>& data);
};

} // namespace nn

#endif // IVFF_H
