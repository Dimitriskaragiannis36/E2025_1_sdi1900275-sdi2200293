#include "lsh.h"
#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>
#include <algorithm>
using namespace std;

namespace nn {

LSH::LSH(int dim, int L, int k, int w, unsigned int seed, DistanceFunc dist_func)
    : dim_(dim), L_(L), k_(k), w_(w), seed_(seed), distance_func_(std::move(dist_func)) {
    tables_.resize(L_); //πιθανόν περιττό
    v_.assign(L_, vector<vector<float>>(k_, vector<float>(dim_, 0.0f)));
    t_.assign(L_, vector<float>(k_, 0.0f));
    r_.assign(L_, vector<uint32_t>(k_, 0));

    //random generators 
    mt19937 rng(seed_); 
    normal_distribution<float> normal_dist(0.0f, 1.0f); //N(0,1) 
    uniform_real_distribution<float> uniform_dist(0.0f, (float)w_); //U[0,w)
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

    if (!distance_func_) {
        //αν δεν δόθηκε custom μετρική, χρησιμοποίησε την default ευκλείδεια
        distance_func_ = [this](const vector<float>& a, const vector<float>& b) {
            return euclidean_distance(a, b);
        };
    }

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

//ID(p) = (Σ r_j * h_j(p)) mod M
uint64_t LSH::compute_id(const vector<long long>& hashes, int table_idx) const {
    uint64_t acc = 0;
    for (int j = 0; j < k_; j++) {
        long long hval = hashes[j] % (long long)M_;
        if (hval < 0) hval += M_;
        acc = (acc + (uint64_t)r_[table_idx][j] * (uint64_t)hval) % M_;
    }
    return acc; // αυτό είναι το ID(p)
}

//g(p) = ID(p) mod TableSize
int LSH::compute_bucket_id(uint64_t id) const {
    return (int)(id % table_size_);
}

//απλή ευκλείδεια απόσταση
float LSH::euclidean_distance(const vector<float>& x, const vector<float>& y) const {
    float dist = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        float diff = x[i] - y[i];
        dist += diff * diff;
    }
    return sqrt(dist);
}

//χτίσιμο index
void LSH::build_index(const vector<vector<float>>& data) {
    data_ptr_ = &data;
    int n = data.size();
    table_size_ = max(1, n / 4);  //ή n/8, n/16 heuristic

    tables_.assign(L_, vector<vector<pair<uint64_t,int>>>(table_size_));

    cout << "Building index on dataset of size " << n << endl;
    for (int id = 0; id < n; id++) {
        const auto& p = data[id];
        for (int i = 0; i < L_; i++) {
            auto hashes = compute_hashes_for_table(p, i);
            uint64_t obj_id = compute_id(hashes, i);
            int bucket = compute_bucket_id(obj_id);
            tables_[i][bucket].push_back({obj_id, id});
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
        uint64_t q_id = compute_id(hashes, i);
        int bucket = compute_bucket_id(q_id);

        //φιλτράρουμε μόνο όσους έχουν ίδιο ID
        for (const auto& [obj_id, idx] : tables_[i][bucket]) {
            if (obj_id == q_id) {
                candidates.insert(idx);
            }
        }
    }

    vector<pair<float,int>> dists;
    dists.reserve(candidates.size());
    for (int id : candidates) {
        float dist = distance_func_(q, (*data_ptr_)[id]);
        dists.emplace_back(dist, id);
    }

    //πιο αποδοτικό: nth_element + sort μόνο των πρώτων k
    if ((int)dists.size() > num_neighbors) {
        nth_element(dists.begin(), dists.begin() + num_neighbors, dists.end());
        dists.resize(num_neighbors);
        sort(dists.begin(), dists.end());
    } else {
        sort(dists.begin(), dists.end());
    }

    vector<int> result;
    result.reserve(min(num_neighbors, (int)dists.size()));
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

//(α) ένας πλησιέστερος γείτονας (ε-approximate NN)
int LSH::nn_query(const vector<float>& q, float epsilon) const {
    auto result = query(q, 1);  
    if (result.empty()) return -1;

    int candidate = result[0];
    float best_dist = distance_func_(q, (*data_ptr_)[candidate]);

    //ψάχνουμε αν υπάρχει άλλος που να κάνει violate το (1+ε)
    for (int i = 0; i < (int)data_ptr_->size(); i++) {
        if (i == candidate) continue;
        float dist = distance_func_(q, (*data_ptr_)[i]);
        if (dist < best_dist / (1 + epsilon)) {
            // τότε ο chosen δεν είναι valid ϵ-NN → update
            candidate = i;
            best_dist = dist;
        }
    }

    return candidate;
}

//(β) N πλησιέστεροι γείτονες
vector<pair<int, float>> LSH::knn_query(const vector<float>& q, int N) const {
    if (!data_ptr_) {
        cerr << "Error: index not built!" << endl;
        return {};
    }

    unordered_set<int> candidates;

    //συλλέγουμε υποψηφίους από όλους τους πίνακες hash
    for (int i = 0; i < L_; ++i) {
        auto hashes = compute_hashes_for_table(q, i);
        uint64_t q_id = compute_id(hashes, i);
        int bucket = compute_bucket_id(q_id);

        for (const auto& [obj_id, idx] : tables_[i][bucket]) {
            if (obj_id == q_id) {
                candidates.insert(idx);
            }
        }
    }

    //υπολογίζουμε αποστάσεις μόνο για τους υποψήφιους
    vector<pair<float,int>> dists;
    dists.reserve(candidates.size());
    for (int id : candidates) {
        float dist = distance_func_(q, (*data_ptr_)[id]);
        dists.emplace_back(dist, id);
    }

    if (dists.empty()) return {};

    //nth_element για top-N
    if ((int)dists.size() > N) {
        nth_element(dists.begin(), dists.begin() + N, dists.end());
        dists.resize(N);
        sort(dists.begin(), dists.end());
    } else {
        sort(dists.begin(), dists.end());
    }

    //επιστρέφουμε (index, απόσταση)
    vector<pair<int,float>> result;
    result.reserve(dists.size());
    for (auto& [dist, id] : dists)
        result.emplace_back(id, dist); //δεν χρειάζεται sqrt ξανά
    return result;
}

//(γ) αναζήτηση εντός ακτίνας R
vector<int> LSH::range_search(const vector<float>& q, float R) const {
    if (!data_ptr_) {
        cerr << "Error: index not built!" << endl;
        return {};
    }

    unordered_set<int> candidates;

    for (int i = 0; i < L_; ++i) {
        auto hashes = compute_hashes_for_table(q, i);
        uint64_t q_id = compute_id(hashes, i);
        int bucket = compute_bucket_id(q_id);

        for (const auto& [obj_id, idx] : tables_[i][bucket]) {
            if (obj_id == q_id) {
                candidates.insert(idx);
            }
        }
    }

    vector<int> result;
    for (int id : candidates) {
        float dist = distance_func_(q, (*data_ptr_)[id]);
        if (dist <= R) {
            result.push_back(id);
        }
    }
    return result;

}


} //namespace nn
