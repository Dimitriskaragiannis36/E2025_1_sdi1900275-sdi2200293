#include "lsh.h"
#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>
#include <algorithm>
using namespace std;

namespace nn {

LSH::LSH(int dim, int L, int k, float w, unsigned int seed)
    : dim_(dim), L_(L), k_(k), w_(w), seed_(seed) {
    tables_.resize(L_); //πιθανόν περιττό
    v_.assign(L_, vector<vector<float>>(k_, vector<float>(dim_, 0.0f)));
    t_.assign(L_, vector<float>(k_, 0.0f));
    r_.assign(L_, vector<uint32_t>(k_, 0));

    //random generators 
    mt19937 rng(seed_); 
    normal_distribution<float> normal_dist(0.0f, 1.0f); //N(0,1) 
    uniform_real_distribution<float> uniform_dist(0.0f, w_); //U[0,w)
    uniform_int_distribution<uint32_t> int_dist(1, M_-1); //για r_j
    
    //γέμισμα v_ t_ και r_ 
    for (int i = 0; i < L_; ++i) { 
        for (int j = 0; j < k_; ++j) { 
            r_[i][j] = int_dist(rng);
            for (int d = 0; d < dim_; ++d) { 
                v_[i][j][d] = normal_dist(rng); 
            } 
            t_[i][j] = uniform_dist(rng); 
        } 
    }

    cout << "LSH initialized with dim=" << dim_
              << ", L=" << L_
              << ", k=" << k_
              << ", w=" << w_
              << endl;
}

//h(p) = floor((p·v + t) / w) για κάθε hash function
vector<long long> LSH::compute_hashes_for_table(const vector<float>& p, int table_idx) const {
    vector<long long> hashes;
hashes.reserve(k_);

for (int j = 0; j < k_; ++j) {
    double dot = 0.0;
    for (int d = 0; d < dim_; ++d) {
        dot += static_cast<double>(p[d]) * static_cast<double>(v_[table_idx][j][d]);
    }
    double val = (dot + t_[table_idx][j]) / w_;
    long long h = static_cast<long long>(floor(val));
    hashes.push_back(h);
}

return hashes;
}

//g(p) = (Σ r_j * h_j(p) mod M) mod TableSize
int LSH::compute_bucket_id(const vector<long long>& hashes, int table_idx) const {
    uint64_t acc = 0;
    for (int j = 0; j < k_; j++) { //για κάθε hash function να μην σκάσει
        long long hval = hashes[j] % (long long)M_;
        if (hval < 0) hval += M_;
        acc = (acc + (uint64_t)r_[table_idx][j] * (uint64_t)hval) % M_;
    }
    return (int)(acc % table_size_);
}


//απλή ευκλείδεια απόσταση
float LSH::euclidean_distance_sq(const vector<float>& x, const vector<float>& y) const {
    float dist = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        float diff = x[i] - y[i];
        dist += diff * diff;
    }
    return dist;
}

//χτίσιμο index
void LSH::build_index(const vector<vector<float>>& data) {
    data_ptr_ = &data;
    int n = data.size();
    table_size_ = max(1, n / 4);  //ή n/8, n/16 heuristic

    tables_.assign(L_, vector<vector<int>>(table_size_));

    cout << "Building index on dataset of size " << n << endl;
    for (int id = 0; id < n; id++) {
        const auto& p = data[id];
        for (int i = 0; i < L_; i++) {
            auto hashes = compute_hashes_for_table(p, i);
            int bucket = compute_bucket_id(hashes, i);
            tables_[i][bucket].push_back(id);
        }
    }
}

//αναζήτηση γειτόνων
vector<int> LSH::query(const vector<float>& q, int num_neighbors) const {
    if (!data_ptr_) {
        cerr << "Error: index not built!" << endl;
        return {};
    }

    unordered_set<int> candidates;

    for (int i = 0; i < L_; ++i) {
        auto hashes = compute_hashes_for_table(q, i);
        int bucket = compute_bucket_id(hashes, i);

        for (int id : tables_[i][bucket]) {
            candidates.insert(id);
        }
    }

    vector<pair<float,int>> dists;
    for (int id : candidates) {
        float dist = euclidean_distance_sq(q, (*data_ptr_)[id]);
        dists.emplace_back(dist, id);
    }

    sort(dists.begin(), dists.end());

    vector<int> result;
    for (int i = 0; i < num_neighbors && i < (int)dists.size(); ++i) {
        result.push_back(dists[i].second);
    }

    return result;
}

//εκκαθάριση index
void LSH::clear_index() {
    for (auto& table : tables_) {
        table.clear();
    }
    cout << "Index cleared." << endl;
}

} //namespace nn
