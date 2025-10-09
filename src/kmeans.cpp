#include "kmeans.h"
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <unordered_set>

namespace clustering {

//κατασκευαστής της κλάσης KMeans
KMeans::KMeans(int k, int max_iters, float tol,
               InitMethod init, unsigned int seed, bool verbose)
    : k_(k),
      max_iters_(max_iters),
      tol_(tol),
      init_method_(init),
      seed_(seed),
      verbose_(verbose) {}


void KMeans::clear() {
    centroids_.clear();
    labels_.clear();
}

//υλοποίηση του αλγορίθμου k-means
float KMeans::squared_distance(const std::vector<float>& a,
                               const std::vector<float>& b) const {
    float dist = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}


//τυχαία αρχικοποίηση των κεντροειδών από τα δεδομένα
void KMeans::init_random(const std::vector<std::vector<float>>& data,
                         std::mt19937& rng) {
    std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
    std::unordered_set<size_t> chosen;

    centroids_.clear();
    while (centroids_.size() < static_cast<size_t>(k_)) {
        size_t idx = dist(rng);
        if (chosen.insert(idx).second) {
            centroids_.push_back(data[idx]);
        }
    }

    if (verbose_) {
        std::cout << "Initialized " << k_ << " centroids randomly." << std::endl;
    }
}

//πιο εξελιγμένη αρχικοποίηση k-means++
void KMeans::init_kmeans_pp(const std::vector<std::vector<float>>& data,
                            std::mt19937& rng) {
    size_t n = data.size();
    size_t d = data[0].size();

    std::uniform_int_distribution<size_t> uni_dist(0, n - 1);
    centroids_.clear();
    centroids_.push_back(data[uni_dist(rng)]); //πρώτο κέντρο τυχαία

    std::vector<float> dist_sq(n, std::numeric_limits<float>::max());

    //επαναλαμβάνουμε μέχρι να έχουμε k κέντρα
    for (int c = 1; c < k_; ++c) {
        //ενημέρωση των αποστάσεων από το κοντινότερο υπάρχον κέντρο
        for (size_t i = 0; i < n; ++i) {
            float d2 = squared_distance(data[i], centroids_.back());
            if (d2 < dist_sq[i])
                dist_sq[i] = d2;
        }

        //επιλογή νέου κέντρου με πιθανότητα D(x)^2
        float sum = 0.0f;
        for (float val : dist_sq)
            sum += val;

        if (sum == 0.0f) {
            //όλα τα σημεία ταυτίζονται — διάλεξε τυχαία
            centroids_.push_back(data[uni_dist(rng)]);
            continue;
        }

        //κανονικοποίηση
        float max_d = *std::max_element(dist_sq.begin(), dist_sq.end());
        if (max_d > 0.0f) {
            for (auto& val : dist_sq)
                val /= max_d;
        }

        //υπολογισμός prefix sums P
        std::vector<float> prefix(n);
        prefix[0] = dist_sq[0];
        for (size_t i = 1; i < n; ++i)
            prefix[i] = prefix[i - 1] + dist_sq[i];

        //τυχαίος αριθμός στο [0, P(n−t)]
        std::uniform_real_distribution<float> prob_dist(0.0f, prefix.back());
        float r = prob_dist(rng);

        //για να αποφύγω πιθανό out-of-range σε rare cases
        r = std::min(r, prefix.back());

        //binary search για να βρούμε r τέτοιο ώστε P(r−1) < x ≤ P(r)
        auto it = std::lower_bound(prefix.begin(), prefix.end(), r);
        size_t next_idx = std::distance(prefix.begin(), it);

        centroids_.push_back(data[next_idx]);

    }

    if (verbose_) {
        std::cout << "Initialized " << k_ << " centroids using K-Means++." << std::endl;
    }
}


//k-means αρχικοποίηση των κεντροειδών
int KMeans::fit(const std::vector<std::vector<float>>& data) {
    if (data.empty()) return 0;
    size_t n = data.size();
    size_t d = data[0].size();
    std::mt19937 rng(seed_);

    //αρχικοποίηση κέντρων
    if (init_method_ == InitMethod::RANDOM)
        init_random(data, rng);
    else
        init_kmeans_pp(data, rng);

    labels_.assign(n, -1);
    std::vector<std::vector<float>> new_centroids(k_, std::vector<float>(d, 0.0f));
    std::vector<int> counts(k_, 0);

    int iter = 0;
    for (; iter < max_iters_; ++iter) {
        bool changed = false;

        //βήμα ανάθεσης
        for (size_t i = 0; i < n; ++i) {
            float best_dist = std::numeric_limits<float>::max();
            int best_cluster = -1;

            for (int c = 0; c < k_; ++c) {
                float dist = squared_distance(data[i], centroids_[c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_cluster = c;
                }
            }

            if (labels_[i] != best_cluster) {
                labels_[i] = best_cluster;
                changed = true;
            }
        }

        //βήμα ενημέρωσης
        for (int c = 0; c < k_; ++c) {
            std::fill(new_centroids[c].begin(), new_centroids[c].end(), 0.0f);
            counts[c] = 0;
        }

        for (size_t i = 0; i < n; ++i) {
            int c = labels_[i];
            counts[c]++;
            for (size_t j = 0; j < d; ++j)
                new_centroids[c][j] += data[i][j];
        }

        for (int c = 0; c < k_; ++c) {
            if (counts[c] > 0) {
                for (size_t j = 0; j < d; ++j)
                    new_centroids[c][j] /= counts[c];
            } else {
                //αν κάποια ομάδα άδειασε, επανατοποθέτησε τυχαία
                std::uniform_int_distribution<size_t> dist(0, n - 1);
                new_centroids[c] = data[dist(rng)];
            }
        }

        //έλεγχος σύγκλισης
        float shift = 0.0f;
        for (int c = 0; c < k_; ++c)
            shift += squared_distance(centroids_[c], new_centroids[c]);

        if (verbose_) {
            std::cout << "Iteration " << iter + 1
                      << ": centroid shift = " << shift << std::endl;
        }

        centroids_ = new_centroids;

        if (shift < tol_) break; //σύγκλιση
        if (!changed) break; //δεν αλλάζουν οι ετικέτες
    }

    if (verbose_) {
        std::cout << "Converged after " << iter + 1 << " iterations." << std::endl;
    }

    return iter + 1;
}

//πρόβλεψη της ομάδας για ένα νέο σημείο
int KMeans::predict(const std::vector<float>& point) const {
    if (centroids_.empty()) {
        std::cerr << "Error: KMeans model not fitted yet!" << std::endl;
        return -1;
    }

    float best_dist = std::numeric_limits<float>::max();
    int best_cluster = -1;

    for (int c = 0; c < k_; ++c) {
        float dist = squared_distance(point, centroids_[c]);
        if (dist < best_dist) {
            best_dist = dist;
            best_cluster = c;
        }
    }

    return best_cluster;
}



} //namespace clustering
