#include "ivff.h"
#include <random>
#include <limits>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <cmath>
#include <iostream>

using namespace std;

namespace nn {

//κονστράκτορας
IVFFlat::IVFFlat(int dim, int nlist, int nprobe, int max_kmeans_iter, unsigned int seed)
    : dim_(dim), nlist_(nlist), nprobe_(nprobe), max_kmeans_iter_(max_kmeans_iter), seed_(seed) {
    centroids_.assign(nlist_, vector<float>(dim_, 0.0f));
    lists_.assign(nlist_, vector<int>());
}

//υπολογισμός τετραγωνικής ευκλείδειας απόστασης
float IVFFlat::euclidean_distance_sq(const vector<float>& a, const vector<float>& b) const {
    float s = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

//τρέξιμο k-means για εύρεση κεντροειδών
void IVFFlat::run_kmeans(const vector<vector<float>>& data) {
    //αρχικοποίηση κεντροειδών με τυχαία δείγματα
    int n = (int)data.size();
    if (n == 0) return;
    std::mt19937 rng(seed_);
    std::uniform_int_distribution<int> dist_idx(0, n-1);
    std::unordered_set<int> picked;
    int tries = 0;
    for (int k = 0; k < nlist_; ++k) {
        int idx = dist_idx(rng);
        while (picked.count(idx) && tries++ < n*10) idx = dist_idx(rng);
        picked.insert(idx);
        centroids_[k] = data[idx]; // copy
    }

    //k-means επαναλήψεις
    vector<int> assignments(n, -1);
    vector<vector<float>> newc(nlist_, vector<float>(dim_, 0.0f));
    vector<int> counts(nlist_, 0);

    for (int iter = 0; iter < max_kmeans_iter_; ++iter) {
        bool changed = false;

        //βήμα ανάθεσης
        for (int i = 0; i < n; ++i) {
            const auto& x = data[i];
            float bestd = numeric_limits<float>::max();
            int bestk = -1;
            for (int k = 0; k < nlist_; ++k) {
                float d = euclidean_distance_sq(x, centroids_[k]);
                if (d < bestd) { bestd = d; bestk = k; }
            }
            if (assignments[i] != bestk) {
                changed = true;
                assignments[i] = bestk;
            }
        }

        //βήμα ενημέρωσης
        for (int k = 0; k < nlist_; ++k) {
            std::fill(newc[k].begin(), newc[k].end(), 0.0f);
            counts[k] = 0;
        }
        for (int i = 0; i < n; ++i) {
            int k = assignments[i];
            if (k < 0) continue;
            const auto& x = data[i];
            for (int d = 0; d < dim_; ++d) newc[k][d] += x[d];
            counts[k] += 1;
        }
        for (int k = 0; k < nlist_; ++k) {
            if (counts[k] > 0) {
                for (int d = 0; d < dim_; ++d) centroids_[k][d] = newc[k][d] / counts[k];
            } else {
                //επανεκκίνηση κεντροειδούς αν δεν έχει δείγματα
                int idx = dist_idx(rng);
                centroids_[k] = data[idx];
            }
        }

        if (!changed) break;
    }
}

void IVFFlat::build_index(const vector<vector<float>>& data) {
    data_ptr_ = &data;
    //τρέξιμο k-means για εύρεση κεντροειδών
    run_kmeans(data);

    //ανάθεση δειγμάτων στις λίστες
    for (auto &lst : lists_) lst.clear();
    int n = (int)data.size();
    for (int i = 0; i < n; ++i) {
        const auto& x = data[i];
        float bestd = numeric_limits<float>::max();
        int bestk = 0;
        for (int k = 0; k < nlist_; ++k) {
            float d = euclidean_distance_sq(x, centroids_[k]);
            if (d < bestd) { bestd = d; bestk = k; }
        }
        lists_[bestk].push_back(i);
    }
    //cout/
    cerr << "IVFFlat: built index with " << nlist_ << " lists, dataset size " << n << "\n";
}

vector<pair<int,float>> IVFFlat::knn_query(const vector<float>& q, int N) const {
    vector<pair<int,float>> empty;
    if (!data_ptr_) return empty;
    //int n = (int)data_ptr_->size();

    //1) βρες τα nprobe_ κοντινότερα κεντροειδή
    vector<pair<float,int>> cent_d;
    cent_d.reserve(nlist_);
    for (int k = 0; k < nlist_; ++k) {
        float d = euclidean_distance_sq(q, centroids_[k]);
        cent_d.emplace_back(d, k);
    }
    if (nprobe_ < nlist_) {
        nth_element(cent_d.begin(), cent_d.begin() + nprobe_, cent_d.end(),
                    [](auto &a, auto &b){ return a.first < b.first; });
        cent_d.resize(nprobe_);
    }
    sort(cent_d.begin(), cent_d.end(), [](auto&a,auto&b){ return a.first < b.first; });

    //2) συγκέντρωσε υποψήφια δείγματα από τις λίστες αυτών των κεντροειδών
    unordered_set<int> candidates;
    for (auto &cd : cent_d) {
        int list_id = cd.second;
        for (int idx : lists_[list_id])
            candidates.insert(idx);
    }

    //3) υπολόγισε τις αποστάσεις από το ερώτημα και επέστρεψε τα N καλύτερα
    vector<pair<float,int>> dists;
    dists.reserve(candidates.size());
    for (int idx : candidates) {
        float dsq = euclidean_distance_sq(q, (*data_ptr_)[idx]);
        dists.emplace_back(dsq, idx);
    }

    if (dists.empty()) return {};
    //κράτησε μόνο τα N καλύτερα
    if ((int)dists.size() > N) {
        nth_element(dists.begin(), dists.begin() + N, dists.end(),
                    [](auto &a, auto &b){ return a.first < b.first; });
        dists.resize(N);
    }
    sort(dists.begin(), dists.end(), [](auto&a,auto&b){ return a.first < b.first; });

    vector<pair<int,float>> result;
    result.reserve(dists.size());
    for (auto &p : dists) result.emplace_back(p.second, sqrt(p.first)); // return true distance
    return result;
}

//εύρεση εντός ακτίνας
vector<int> IVFFlat::range_search(const vector<float>& q, float R) const {
    vector<int> result;
    if (!data_ptr_) return result;
    float R2 = R * R;

    //τσέκαρε τα nprobe_ κοντινότερα κεντροειδή
    vector<pair<float,int>> cent_d;
    cent_d.reserve(nlist_);
    for (int k = 0; k < nlist_; ++k) {
        float d = euclidean_distance_sq(q, centroids_[k]);
        cent_d.emplace_back(d, k);
    }
    if (nprobe_ < nlist_) {
        nth_element(cent_d.begin(), cent_d.begin() + nprobe_, cent_d.end(),
                    [](auto &a, auto &b){ return a.first < b.first; });
        cent_d.resize(nprobe_);
    }
    sort(cent_d.begin(), cent_d.end(), [](auto&a,auto&b){ return a.first < b.first; });
    //συγκέντρωσε δείγματα από τις λίστες αυτών των κεντροειδών
    unordered_set<int> visited;
    for (auto &cd : cent_d) {
        int list_id = cd.second;
        for (int idx : lists_[list_id]) {
            if (visited.insert(idx).second) {
                float dsq = euclidean_distance_sq(q, (*data_ptr_)[idx]);
                if (dsq <= R2) result.push_back(idx);
            }
        }
    }
    return result;
}

//καθαρισμός ευρετηρίου
void IVFFlat::clear_index() {
    for (auto &lst : lists_) lst.clear();
    for (auto &c : centroids_) std::fill(c.begin(), c.end(), 0.0f);
    data_ptr_ = nullptr;
}

} //namespace nn
