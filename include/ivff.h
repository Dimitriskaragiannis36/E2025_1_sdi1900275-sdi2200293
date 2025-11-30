#ifndef IVFF_H
#define IVFF_H

#include "helper.h" //utils::DistanceFunc
#include "kmeans.h" //kmeans::KMeans
#include "silhouette.h" //clustering::Silhouette
#include <vector> //std::vector
#include <limits> //std::numeric_limits
#include <utility> //std::pair

namespace ivf {

//IVF-Flat Index
class IVFFlat {
public: //κονστράκτορας με τιμές ώστε να φτιάξει τα αντικείμενα
    IVFFlat(int kclusters = 4,  //αριθμός clusters
            int nprobe = 5,   //αριθμός clusters που θα εξεταστούν
            unsigned int seed = 1,  //σπόρος RNG
            int N = 1,          //default N για knn_query 
            float R = 2000.0f,  //default R για range_search
            utils::DistanceFunc dist_func = nullptr,   // συνάρτηση απόστασης
            int mode = 0);           // <-- ΝΕΑ ΠΑΡΑΜΕΤΡΟΣ (0=normal IVF, 1=new mode)

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
    int nlist() const { return  kclusters_; } //αριθμός clusters
    int nprobe() const { return nprobe_; }  //αριθμός clusters που εξετάζονται
    const std::vector<std::vector<float>>& centroids() const { return centroids_; } //centroids
    const std::vector<std::vector<float>>& data() const { return *data_ptr_; }  //δεδομένα
    utils::DistanceFunc distance_func() const { return dist_func_; }  //συνάρτηση απόστασης

private:
    int kclusters_;    //αριθμός clusters
    int nprobe_;       //αριθμός clusters που εξετάζονται
    unsigned int seed_;  //σπόρος RNG
    int N_;            //default N για KNN
    float R_;          //default R για range search
    utils::DistanceFunc dist_func_;

    // <-- ΝΕΑ ΜΕΤΑΒΛΗΤΗ ΓΙΑ mode
    int mode_;   // 0 = IVF Flat, 1 = νέο mode (ό,τι υλοποιήσεις)

    std::vector<std::vector<float>> centroids_;     //C = {c1, ..., ck}
    std::vector<std::vector<int>> inverted_lists_;  //IL_j = ids των points
    const std::vector<std::vector<float>>* data_ptr_; //pointer στα data

};

void compute_and_output_knn_graph(ivf::IVFFlat& index,
                                  const std::vector<std::vector<float>>& data,
                                  int k,
                                  const std::string& dataset_name);


} //namespace ivf

#endif //IVFF_H
