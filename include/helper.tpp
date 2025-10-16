#pragma once
#include <chrono>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <unordered_set>

namespace utils
{

    /*παράμετροι για εκτέλεση ερωτημάτων*/
    template <typename IndexType>
    void run_queries(IndexType &index,
                     const std::vector<std::vector<float>> &data,
                     const std::vector<std::vector<float>> &queries,
                     const Params &params,
                     std::ofstream &out)
    {
        if (params.use_lsh)
            out << "LSH" << std::endl;
        else if (params.use_ivfflat)
            out << "IVFFlat" << std::endl;
        /*else if (params.use_ivfpq)
            out << "IVFPQ" << std::endl;*/
        else if (params.use_hypercube)
            out
                << "Hypercube" << std::endl;
        else
            out << "UnknownMethod" << std::endl;

        /*μετρήσεις χρόνου*/
        using clock = std::chrono::high_resolution_clock;

        auto t0_total = clock::now();
        double sum_AF = 0.0, sum_recall = 0.0, sum_tApprox = 0.0, sum_tTrue = 0.0;
        /*εκτέλεση ερωτημάτων*/
        for (size_t qi = 0; qi < queries.size(); ++qi)
        {
            out << "Query: " << qi << "\n";

            auto t0 = clock::now();
            auto knn = index.knn_query(queries[qi], params.N);
            auto t1 = clock::now();
            double tApprox_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            sum_tApprox += tApprox_ms;

            /*true distances*/
            auto t0_true = clock::now();
            std::vector<std::pair<double, int>> true_scores;
            true_scores.reserve(data.size());
            for (size_t i = 0; i < data.size(); ++i)
            {
                double dist = 0.0;
                for (size_t d = 0; d < queries[qi].size(); ++d)
                    dist += (queries[qi][d] - data[i][d]) * (queries[qi][d] - data[i][d]);
                true_scores.emplace_back(std::sqrt(dist), (int)i);
            }
            /*εύρεση των N μικρότερων αποστάσεων*/
            int denom_N = std::min(params.N, (int)true_scores.size());
            std::nth_element(true_scores.begin(), true_scores.begin() + denom_N, true_scores.end(),
                             [](auto &a, auto &b)
                             { return a.first < b.first; });
            true_scores.resize(denom_N);
            std::sort(true_scores.begin(), true_scores.end(),
                      [](auto &a, auto &b)
                      { return a.first < b.first; });
            /*αποθήκευση των αποτελεσμάτων*/
            std::vector<int> true_ids;
            std::vector<double> true_dists;
            for (auto &p : true_scores)
            {
                true_ids.push_back(p.second);
                true_dists.push_back(p.first);
            }
            /*χρόνος υπολογισμού true distances*/
            auto t1_true = clock::now();
            double tTrue_ms = std::chrono::duration<double, std::milli>(t1_true - t0_true).count();
            sum_tTrue += tTrue_ms;

            /*εκτύπωση*/
            std::vector<double> approx_dists;
            int nn_counter = 1;
            for (auto [idx, dist] : knn)
            {
                approx_dists.push_back(dist);
                double trueDist = -1.0;
                auto it = std::find(true_ids.begin(), true_ids.end(), idx);
                if (it != true_ids.end())
                {
                    trueDist = true_dists[it - true_ids.begin()];
                }
                else
                {
                    double dsum = 0.0;
                    for (size_t d = 0; d < queries[qi].size(); ++d)
                        dsum += (queries[qi][d] - data[idx][d]) * (queries[qi][d] - data[idx][d]);
                    trueDist = std::sqrt(dsum);
                }
                out << "Nearest neighbor-" << nn_counter++ << ": " << idx << "\n";
                out << "distanceApproximate: " << dist << "\n";
                out << "distanceTrue: " << trueDist << "\n";
            }
            /*υπολογισμός AF και Recall@N*/
            if (!approx_dists.empty() && !true_dists.empty() && true_dists[0] > 1e-12)
                sum_AF += approx_dists[0] / true_dists[0];
            /*Recall@N*/
            std::unordered_set<int> true_set(true_ids.begin(), true_ids.end());
            int overlap = 0;
            for (auto [id, d] : knn)
                if (true_set.count(id))
                    overlap++;
            sum_recall += (denom_N > 0) ? (double)overlap / denom_N : 0.0;
            /*εκτύπωση χρόνων*/
            if (params.do_range)
            {
                auto range = index.range_search(queries[qi], params.R); /*generic call*/
                out << "R-near neighbors:\n";
                for (int idx : range)
                    out << idx << "\n";
            }
            out << "\n";
        }

        auto t1_total = clock::now();
        double total_sec = std::chrono::duration<double>(t1_total - t0_total).count();
        double QPS = queries.empty() ? 0.0 : queries.size() / total_sec;

        out << "Average AF: " << (sum_AF / queries.size()) << "\n";
        out << "Recall@N: " << (sum_recall / queries.size()) << "\n";
        out << "QPS: " << QPS << "\n";
        out << "tApproximateAverage: " << (sum_tApprox / queries.size()) << " ms\n";
        out << "tTrueAverage: " << (sum_tTrue / queries.size()) << " ms\n";
    }

    /*nn*/
    template <typename IndexType>
    int nn_query(const IndexType &index, const std::vector<float> &q, float epsilon)
    {
        auto candidates = index.query_candidates(q);
        const auto &data = index.data();
        auto dist_func = index.distance_func();

        if (candidates.empty())
            return -1;

        int best_id = -1;
        float best_dist = std::numeric_limits<float>::max();
        /*ακριβής αναζήτηση μεταξύ των υποψηφίων*/
        for (int id : candidates)
        {
            float dist = dist_func(q, data[id]);
            if (dist < best_dist)
            {
                best_dist = dist;
                best_id = id;
            }
        }

        /*ε-approximate check (προαιρετικό refinement)*/
        for (int i = 0; i < (int)data.size(); ++i)
        {
            float dist = dist_func(q, data[i]);
            if (dist < best_dist / (1 + epsilon))
            {
                best_dist = dist;
                best_id = i;
            }
        }
        return best_id;
    }

    /*knn*/
    template <typename IndexType>
    std::vector<std::pair<int, float>> knn_query(const IndexType &index,
                                                 const std::vector<float> &q,
                                                 int N)
    {
        auto candidates = index.query_candidates(q);
        const auto &data = index.data();
        auto dist_func = index.distance_func();
        /*υπολογισμός αποστάσεων*/
        std::vector<std::pair<float, int>> dists;
        dists.reserve(candidates.size());
        for (int id : candidates)
            dists.emplace_back(dist_func(q, data[id]), id);

        if ((int)dists.size() > N)
            std::nth_element(dists.begin(), dists.begin() + N, dists.end());
        else
            N = dists.size();

        std::sort(dists.begin(), dists.begin() + N);
        /*αποθήκευση αποτελεσμάτων*/
        std::vector<std::pair<int, float>> result;
        result.reserve(N);
        /*μετατροπή σε (id, dist)*/
        for (int i = 0; i < N; ++i)
            result.emplace_back(dists[i].second, dists[i].first);
        return result;
    }

    /*range*/
    template <typename IndexType>
    std::vector<int> range_search(const IndexType &index,
                                  const std::vector<float> &q,
                                  float R)
    {
        auto candidates = index.query_candidates(q);
        const auto &data = index.data();
        auto dist_func = index.distance_func();
        /*υπολογισμός αποστάσεων*/
        std::vector<int> result;
        for (int id : candidates)
        {
            float dist = dist_func(q, data[id]);
            if (dist <= R)
                result.push_back(id);
        }
        return result;
    }

} /*namespace utils*/
