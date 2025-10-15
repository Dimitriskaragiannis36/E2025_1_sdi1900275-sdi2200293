#include "hypercube.h"
#include <algorithm>
#include <bitset>
#include <random>
#include <unordered_set>
#include <queue>
#include <iostream>

namespace hcube
{

    /*αποθηκεύει τις βασικές παραμέτρους (διαστάσεις, seed, μέγιστα κτλ.)*/
    Hypercube::Hypercube(int dim, int dprime, int w, int max_candidates, int max_probes,
                         unsigned int seed, utils::DistanceFunc dist_func)
        : dim_(dim),
          dprime_(dprime),
          w_(w),
          max_candidates_(max_candidates),
          max_probes_(max_probes),
          seed_(seed),
          dist_func_(dist_func) {}

    /*Αντιστοιχίζει την τιμή h_i(p) -> bit {0,1} και δημιουργεί/ενημερώνει τη bit_map_ για συνεπή ανάθεση. Επιστρέφει τον 64-bit δείκτη κορυφής (vertex id).*/
    uint64_t Hypercube::point_to_vertex(const std::vector<float> &p) const
    {
        uint64_t vertex = 0;

        for (int i = 0; i < dprime_; ++i)
        {
            float dot = 0.0f;
            /*υπολογισμός εσωτερικού γινομένου v_i · p*/
            for (int j = 0; j < dim_; ++j)
                dot += v_[i][j] * p[j];

            /*υπολογισμός του h_i(p) = floor((v_i·p + t_i)/w)*/
            long long h = static_cast<long long>(std::floor((dot + t_[i]) / w_));

            /*αν η τιμή h έχει ξαναεμφανιστεί, χρησιμοποιεί το αποθηκευμένο bit*/
            auto it = bit_map_[i].find(h);
            uint8_t bit;
            if (it != bit_map_[i].end())
            {
                bit = it->second;
            }
            else
            {
                /*δημιουργία ντετερμινιστικού bit {0,1} με χρήση hash + seed ώστε να είναι επαναλήψιμο*/
                std::mt19937 gen(seed_ ^ (i * 1315423911u) ^ (std::hash<long long>{}(h)));
                bit = gen() % 2;
                bit_map_[i][h] = bit;
            }

            // Προσθήκη του bit στη δυαδική αναπαράσταση του vertex id
            vertex |= (static_cast<uint64_t>(bit) << i);
        }

        return vertex;
    }

    /*Δημιουργία του ευρετηρίου:
      - Δημιουργεί τυχαίες προβολές (v_, t_)
      - Υπολογίζει για κάθε σημείο σε ποια κορυφή (vertex) ανήκει
      - Αποθηκεύει το index του σημείου στον αντίστοιχο κάδο*/
    void Hypercube::build_index(const std::vector<std::vector<float>> &data)
    {
        data_ptr_ = &data;

        std::mt19937 gen(seed_);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::uniform_real_distribution<float> uniform(0.0f, w_);

        /*δημιουργία τυχαίων διανυσμάτων προβολής και μετατοπίσεων*/
        v_.assign(dprime_, std::vector<float>(dim_));
        t_.assign(dprime_, 0.0f);
        bit_map_.assign(dprime_, {});

        for (int i = 0; i < dprime_; ++i)
        {
            for (int j = 0; j < dim_; ++j)
                v_[i][j] = normal(gen);
            t_[i] = uniform(gen);
        }

        /*καθαρισμός προηγούμενων κάδων*/
        vertex_buckets_.clear();

        /*αντιστοίχιση κάθε σημείου σε κορυφή*/
        for (int idx = 0; idx < (int)data.size(); ++idx)
        {
            uint64_t vertex = point_to_vertex(data[idx]);
            vertex_buckets_[vertex].push_back(idx);
        }
    }

    /*αφαίρεση όλων των δεδομένων*/
    void Hypercube::clear_index()
    {
        vertex_buckets_.clear();
        bit_map_.clear();
        v_.clear();
        t_.clear();
    }

    /*Συλλογή υποψήφιων σημείων από κοντινές κορυφές του hypercube.
      Διασχίζει κορυφές με BFS, αυξάνοντας προοδευτικά την απόσταση Hamming.*/
    void Hypercube::collect_neighbors_by_hamming(uint64_t vertex, std::vector<int> &out_candidates) const
    {
        std::queue<uint64_t> q;
        std::unordered_set<uint64_t> visited;
        q.push(vertex);
        visited.insert(vertex);

        int probes = 0;
        while (!q.empty() && probes < max_probes_)
        {
            uint64_t v = q.front();
            q.pop();
            ++probes;

            /*αν υπάρχει bucket για αυτήν την κορυφή, προσθέτει τα indices*/
            auto it = vertex_buckets_.find(v);
            if (it != vertex_buckets_.end())
            {
                for (int idx : it->second)
                    out_candidates.push_back(idx);
                if ((int)out_candidates.size() >= max_candidates_)
                    return;
            }

            /*δημιουργία γειτόνων με Hamming απόσταση 1*/
            for (int b = 0; b < dprime_; ++b)
            {
                uint64_t neighbor = v ^ (1ULL << b);
                if (!visited.count(neighbor))
                {
                    visited.insert(neighbor);
                    q.push(neighbor);
                }
            }
        }
    }

    /*επιστρέφει όλους τους υποψηφίους γείτονες για ένα ερώτημα q*/
    std::vector<int> Hypercube::query_candidates(const std::vector<float> &q) const
    {
        uint64_t vertex = point_to_vertex(q);
        std::vector<int> candidates;
        collect_neighbors_by_hamming(vertex, candidates);
        return candidates;
    }

    /*Εκτέλεση ερωτήματος k-NN:
      - Παίρνει όλους τους υποψηφίους (από κοντινές κορυφές)
      - Υπολογίζει αποστάσεις
      - Κρατά τους Ν μικρότερους*/
    std::vector<std::pair<int, float>> Hypercube::knn_query(const std::vector<float> &q, int N) const
    {
        std::vector<int> candidates = query_candidates(q);
        std::vector<std::pair<int, float>> results;

        for (int idx : candidates)
        {
            float dist = dist_func_(q, (*data_ptr_)[idx]);
            results.emplace_back(idx, dist);
        }

        /*ταξινόμηση(μερική) με βάση την απόσταση*/
        std::partial_sort(results.begin(), results.begin() + std::min(N, (int)results.size()),
                          results.end(), [](auto &a, auto &b)
                          { return a.second < b.second; });

        if ((int)results.size() > N)
            results.resize(N);
        return results;
    }

    /*range search: επιστρέφει όλα τα σημεία με απόσταση ≤ R*/
    std::vector<int> Hypercube::range_search(const std::vector<float> &q, float R) const
    {
        std::vector<int> candidates = query_candidates(q);
        std::vector<int> in_range;
        for (int idx : candidates)
        {
            float dist = dist_func_(q, (*data_ptr_)[idx]);
            if (dist <= R)
                in_range.push_back(idx);
        }
        return in_range;
    }

} /*namespace hcube*/
