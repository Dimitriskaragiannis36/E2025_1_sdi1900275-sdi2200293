#include "silhouette.h"
#include <limits>
#include <numeric>
#include <iostream>

using namespace std;

namespace clustering {

Silhouette::Silhouette(utils::DistanceFunc dist_func)
    : dist_func_(std::move(dist_func)) {}


std::vector<float> Silhouette::compute_per_point(
    const std::vector<std::vector<float>>& data,
    const std::vector<int>& labels,
    int k
) const {
    size_t n = data.size();
    std::vector<float> s_values(n, 0.0f);

    //ομαδοποιούμε τα σημεία ανά cluster για αποδοτικότητα
    std::vector<std::vector<size_t>> clusters(k);
    for (size_t i = 0; i < n; ++i)
        clusters[labels[i]].push_back(i);

    for (size_t i = 0; i < n; ++i) {
        int cluster_i = labels[i];
        const auto& same_cluster = clusters[cluster_i];

        //υπολογισμός a(i): μέση απόσταση στο ίδιο cluster
        float a_i = 0.0f;
        if (same_cluster.size() > 1) {
            for (size_t j_idx : same_cluster) {
                if (j_idx == i) continue;
                a_i += dist_func_(data[i], data[j_idx]);
            }
            a_i /= (same_cluster.size() - 1);
        } else {
            a_i = 0.0f; //μόνο του το σημείο
        }

        //υπολογισμός b(i): ελάχιστη μέση απόσταση σε άλλο cluster
        float b_i = std::numeric_limits<float>::max();
        for (int c = 0; c < k; ++c) {
            if (c == cluster_i || clusters[c].empty()) continue;
            float avg_dist = 0.0f;
            for (size_t j_idx : clusters[c])
                avg_dist += dist_func_(data[i], data[j_idx]);
            avg_dist /= clusters[c].size();

            if (avg_dist < b_i)
                b_i = avg_dist;
        }

        //υπολογισμός silhouette s(i)
        float s_i = 0.0f;
        if (a_i == 0.0f && b_i == 0.0f)
            s_i = 0.0f;
        else if (a_i < b_i)
            s_i = 1.0f - (a_i / b_i);
        else if (a_i > b_i)
            s_i = (b_i / a_i) - 1.0f;
        else
            s_i = 0.0f;

        s_values[i] = s_i;
    }

    return s_values;
}


float Silhouette::compute(
    const std::vector<std::vector<float>>& data,
    const std::vector<int>& labels,
    int k
) const {
    auto s_values = compute_per_point(data, labels, k);
    float sum = std::accumulate(s_values.begin(), s_values.end(), 0.0f);
    return s_values.empty() ? 0.0f : sum / s_values.size();
}

} //namespace clustering
