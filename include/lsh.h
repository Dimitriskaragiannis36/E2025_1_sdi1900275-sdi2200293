#ifndef LSH_H
#define LSH_H

#include "helper.h"
#include <vector>
#include <cstdint>

using namespace std;

namespace nn {

class LSH {
public:
    //κονστράκτορας με βασικές παραμέτρους
    LSH(int dim, int L = 5, int k = 4, int w = 4, unsigned int seed = 12345,
        utils::DistanceFunc dist_func = nullptr);

    //χτίσιμο index πάνω σε dataset
    void build_index(const vector<vector<float>>& data);

    //γενική query συνάρτηση που χρησιμοποιούν οι παρακάτω
    vector<int> query(const vector<float>& q, int num_neighbors = 10) const;

    //(α) ένας πλησιέστερος γείτονας
    int nn_query(const vector<float>& q, float epsilon = 0.1f) const;

    //(β) N πλησιέστεροι γείτονες
    vector<pair<int, float>> knn_query(const vector<float>& q, int N) const;

    //(γ) αναζήτηση εντός ακτίνας R
    vector<int> range_search(const vector<float>& q, float R) const;

    //εκκαθάριση index
    void clear_index();

private:
    int dim_;   //διάσταση
    int L_;     //αριθμός hash tables L
    int k_;     //αριθμός hash functions ανά πίνακα k
    int w_;   //μέγεθος παραθύρου w
    unsigned int seed_;
    utils::DistanceFunc distance_func_;

    static constexpr uint64_t M_ = 4294967291ULL; //μεγάλο prime κοντά στο 2^32 
    //για ↑ συγκρούσεις ↑ w ή ↓ k ενώ ↓ συγκρούσεις ↓ w ή ↑ k
    int table_size_ = 1; //TableSize = n/4 ή n/8, ορίζεται στο build_index

    //κάθε bucket: λίστα (ID, index)
    vector<vector<vector<pair<uint64_t,int>>>> tables_;

    //τυχαία διανύσματα v ∼ N(0,1)^d (L × k × d) 
    vector<vector<vector<float>>> v_;

    //μετατοπίσεις t ∼ U[0,w) (L × k) 
    vector<vector<float>> t_;
    
    //random coefficients r (L × k)
    vector<vector<uint32_t>> r_;

    //δείκτης στα δεδομένα
    const vector<vector<float>>* data_ptr_ = nullptr;

    //h(p) = floor((p·v + t) / w) για κάθε hash function 
    vector<long long> compute_hashes_for_table(const vector<float>& p, int table_idx) const;

    //ID(p) = (Σ r_j * h_j(p)) mod M
    uint64_t compute_id(const vector<long long>& hashes, int table_idx) const;

    //g(p) = ID(p) mod TableSize
    int compute_bucket_id(uint64_t id) const;

};

} //namespace nn

#endif //LSH_H

