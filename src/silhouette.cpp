#include "silhouette.h"
#include <limits>
#include <numeric>
#include <iostream>

namespace clustering {

//κονστρακτορας που δέχεται τη συνάρτηση απόστασης
Silhouette::Silhouette(utils::DistanceFunc dist_func)
    : dist_func_(std::move(dist_func)) {}

//υπολογίζει και επιστρέφει τα επιμέρους s(i)
std::vector<float> Silhouette::compute_per_point(
    const std::vector<std::vector<float>>& data,
    const std::vector<int>& labels,
    int k
) const {
    std::size_t n = data.size(); //αριθμός σημείων
    std::vector<float> s_values(n, 0.0f); //αποθήκευση s(i)

    //ομαδοποιούμε τα σημεία ανά cluster για αποδοτικότητα
    std::vector<std::vector<std::size_t>> clusters(k);
    for (std::size_t i = 0; i < n; ++i)
        clusters[labels[i]].push_back(i); //αποθήκευση index σημείου

    for (std::size_t i = 0; i < n; ++i) {
        int cluster_i = labels[i]; //cluster του σημείου i
        const auto& same_cluster = clusters[cluster_i]; //σημεία στο ίδιο cluster

        //υπολογισμός a(i): μέση απόσταση στο ίδιο cluster
        float a_i = 0.0f;
        if (same_cluster.size() > 1) {
            for (std::size_t j_idx : same_cluster) {
                if (j_idx == i) continue; //παράλειψη του ίδιου σημείου
                a_i += dist_func_(data[i], data[j_idx]); //απόσταση μεταξύ i και j
            }
            a_i /= (same_cluster.size() - 1); //μέσος όρος απόστασης
        } else {
            a_i = 0.0f; //μόνο του το σημείο
        }

        //υπολογισμός b(i): ελάχιστη μέση απόσταση σε άλλο cluster
        float b_i = std::numeric_limits<float>::max();
        for (int c = 0; c < k; ++c) {
            if (c == cluster_i || clusters[c].empty()) continue;
            float avg_dist = 0.0f; //μέση απόσταση στο cluster c
            for (std::size_t j_idx : clusters[c])
                avg_dist += dist_func_(data[i], data[j_idx]); //απόσταση μεταξύ i και j
            avg_dist /= clusters[c].size(); //μέσος όρος απόστασης

            if (avg_dist < b_i)
                b_i = avg_dist;
        }

        //υπολογισμός silhouette s(i)
        float s_i = 0.0f; //αρχικοποίηση s(i)
        if (a_i == 0.0f && b_i == 0.0f)
            s_i = 0.0f; //ορισμός για μοναδικό σημείο
        else if (a_i < b_i)
            s_i = 1.0f - (a_i / b_i); //καλό clustering
        else if (a_i > b_i)
            s_i = (b_i / a_i) - 1.0f; //κακό clustering
        else
            s_i = 0.0f; //a_i == b_i

        s_values[i] = s_i; //αποθήκευση s(i)
    }

    return s_values;
}

//υπολογίζει το συνολικό μέσο silhouette score
float Silhouette::compute(
    const std::vector<std::vector<float>>& data,
    const std::vector<int>& labels,
    int k
) const {
    auto s_values = compute_per_point(data, labels, k); //λήψη s(i)
    float sum = std::accumulate(s_values.begin(), s_values.end(), 0.0f); //άθροιση s(i)
    return s_values.empty() ? 0.0f : sum / s_values.size(); //μέσος όρος
}

//υπολογίζει και επιστρέφει τα μέση silhouette score ανά cluster
std::vector<float> Silhouette::compute_per_cluster(
    const std::vector<std::vector<float>>& data,
    const std::vector<int>& labels,
    int k
) const {
    auto s_values = compute_per_point(data, labels, k); //λήψη s(i)
    std::vector<float> cluster_avg(k, 0.0f); //αποθήκευση μέσων όρων ανά cluster
    std::vector<int> cluster_count(k, 0); //αποθήκευση πλήθους σημείων ανά cluster

    //συγκεντρώνουμε τα s(i) ανά cluster
    for (std::size_t i = 0; i < labels.size(); ++i) {
        int c = labels[i]; //cluster του σημείου i
        if (c >= 0 && c < k) {
            cluster_avg[c] += s_values[i]; //πρόσθεση s(i) στο cluster
            cluster_count[c]++; //αύξηση πλήθους σημείων στο cluster
        }
    }

    //υπολογίζουμε τον μέσο όρο ανά cluster
    for (int c = 0; c < k; ++c) {
        if (cluster_count[c] > 0) {
            cluster_avg[c] /= cluster_count[c]; //μέσος όρος s(i) για το cluster
        } else {
            //undefined silhouette για άδειο cluster
            cluster_avg[c] = std::numeric_limits<float>::quiet_NaN(); //ή κάποιο άλλο σήμα
            std::cerr << "Warning: Empty cluster " << c << " (silhouette undefined)\n";
        }
    }

    return cluster_avg;
}

} //namespace clustering
