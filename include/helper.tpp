#pragma once
#include <chrono> //για μέτρηση χρόνου
#include <iostream> //για έξοδο
#include <algorithm> //για sort, nth_element
#include <unordered_set> //για recall υπολογισμό

namespace utils
{

    //τελική έκδοση run_queries: precompute brute-force true results once
    template <typename IndexType>
    void run_queries(IndexType &index,
                    const std::vector<std::vector<float>> &data,
                    const std::vector<std::vector<float>> &queries,
                    const Params &params,
                    std::ofstream &out)
    {
        if (params.use_lsh) //LSH
            out << "LSH" << std::endl;
        else if (params.use_ivfflat) //IVFFlat
            out << "IVFFlat" << std::endl;
        else if (params.use_ivfpq) //IVFPQ
            out << "IVFPQ" << std::endl; 
        else if (params.use_hypercube) //Hypercube
            out << "Hypercube" << std::endl;
        else //άλλο
            out << "UnknownMethod" << std::endl;

        using clock = std::chrono::high_resolution_clock; //χρονόμετρο υψηλής ακρίβειας

        //προϋπολογισμός (brute-force) true distances ΜΙΑ ΦΟΡΑ για όλα τα queries
        auto t0_true_all = clock::now();

        //αποθηκεύουμε για κάθε query τα top-N true (ids + distances)
        std::vector<std::vector<int>> true_ids_all;
        std::vector<std::vector<double>> true_dists_all;
        true_ids_all.resize(queries.size());
        true_dists_all.resize(queries.size());

        for (size_t qi = 0; qi < queries.size(); ++qi) {
            const auto &q = queries[qi];
            //υπολογισμός αποστάσεων σε όλα τα σημεία (bruteforce)
            std::vector<std::pair<double,int>> true_scores;
            true_scores.reserve(data.size()); //απόσταση, id
            for (size_t i = 0; i < data.size(); ++i) {
                double dist = 0.0; //ευκλείδεια απόσταση
                for (size_t d = 0; d < q.size(); ++d) 
                    dist += (q[d] - data[i][d]) * (q[d] - data[i][d]); //τετράγωνο απόστασης
                true_scores.emplace_back(std::sqrt(dist), (int)i); //αποθήκευση (απόσταση, id)
            }

            //επιλέγουμε τα N μικρότερα (ή όσο υπάρχουν)
            int denom_N = std::min(params.N, (int)true_scores.size());
            if (denom_N > 0) { //μη κενό σύνολο δεδομένων
                std::nth_element(true_scores.begin(), true_scores.begin() + denom_N, true_scores.end(),
                                [](auto &a, auto &b){ return a.first < b.first; }); //μερική ταξινόμηση
                true_scores.resize(denom_N); //κρατάμε μόνο τα top-N
                std::sort(true_scores.begin(), true_scores.end(),
                        [](auto &a, auto &b){ return a.first < b.first; }); //ταξινόμηση των top-N
                //αποθηκευση ids + distances
                true_ids_all[qi].reserve(true_scores.size()); 
                true_dists_all[qi].reserve(true_scores.size());
                for (auto &p : true_scores) { //αποθήκευση
                    true_dists_all[qi].push_back(p.first);
                    true_ids_all[qi].push_back(p.second);
                }
            } else {
                //κενό σύνολο δεδομένων
                true_ids_all[qi].clear();
                true_dists_all[qi].clear();
            }
        }

        auto t1_true_all = clock::now(); //τέλος precompute true results
        double tTrueTotal_ms = std::chrono::duration<double, std::milli>(t1_true_all - t0_true_all).count(); //ms
        double tTrueAverage_ms = queries.empty() ? 0.0 : tTrueTotal_ms / queries.size(); //ms ανά query

        //εκτέλεση queries με χρήση του index (approx) και χρήση των precomputed true results
        auto t0_total = clock::now(); //χρόνος έναρξης συνολικής εκτέλεσης  
        double sum_AF = 0.0; //για Average Fraction
        double sum_recall = 0.0; //για Recall@N
        double sum_tApprox = 0.0; //για μέσο χρόνο approx ανά query

        for (size_t qi = 0; qi < queries.size(); ++qi) { //για κάθε query
            out << "Query: " << qi << "\n";

            //approximate knn από index
            auto t0 = clock::now(); //χρόνος έναρξης approx
            auto knn = index.knn_query(queries[qi], params.N); //εκτέλεση knn query
            auto t1 = clock::now(); //τέλος approx
            double tApprox_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(); //ms
            sum_tApprox += tApprox_ms; //συσσώρευση χρόνου approx

            //ανακτούμε τα precomputed true αποτελέσματα για αυτό το query
            const auto &true_ids = true_ids_all[qi];    //ids
            const auto &true_dists = true_dists_all[qi]; //distances

            //εκτύπωση αποτελεσμάτων approx + true (αν χρειάζεται υπολογισμός trueDist για μη-topN)
            std::vector<double> approx_dists; //για Average Fraction
            int nn_counter = 1;  //μετρητής γειτόνων
            for (auto [idx, dist] : knn) {
                approx_dists.push_back(dist); //αποθήκευση απόστασης approx
                double trueDist = -1.0;
                auto it = std::find(true_ids.begin(), true_ids.end(), idx); //αναζήτηση idx στα true top-N
                if (it != true_ids.end()) {
                    trueDist = true_dists[it - true_ids.begin()]; //αν βρέθηκε, παίρνουμε την true απόσταση
                } else {
                    //αν το idx δεν ήταν στα top-N true, υπολογίσουμε την ακριβή απόσταση (μόνο για εμφάνιση)
                    double dsum = 0.0;
                    for (size_t d = 0; d < queries[qi].size(); ++d) //ευκλείδεια απόσταση
                        dsum += (queries[qi][d] - data[idx][d]) * (queries[qi][d] - data[idx][d]);
                    trueDist = std::sqrt(dsum); //υπολογισμός true απόστασης
                }
                out << "Nearest neighbor-" << nn_counter++ << ": " << idx << "\n";
                out << "distanceApproximate: " << dist << "\n";
                out << "distanceTrue: " << trueDist << "\n";
            }

            //υπολογισμός AF (Average Fraction) — χρησιμοποιούμε το πρώτο approx και πρώτο true
            if (!approx_dists.empty() && !true_dists.empty() && true_dists[0] > 1e-12)
                sum_AF += approx_dists[0] / true_dists[0]; //συσσώρευση AF

            //Recall@N: πόσοι από τους true top-N υπάρχουν στα approx αποτελέσματα
            std::unordered_set<int> true_set(true_ids.begin(), true_ids.end());
            int overlap = 0; //μετρητής επικαλύψεων
            for (auto [id, d] : knn)
                if (true_set.count(id)) overlap++; //αν το id είναι στα true top-N
            int denom_N = std::min(params.N, (int)true_ids.size()); //κανονικοποίηση
            sum_recall += (denom_N > 0) ? (double)overlap / denom_N : 0.0; //συσσώρευση recall

            //range search (αν ζητείται)
            if (params.do_range) { 
                auto range = index.range_search(queries[qi], params.R); //εκτέλεση range search
                out << "R-near neighbors:\n";
                for (int idx : range) //εκτύπωση αποτελεσμάτων range
                    out << idx << "\n";
            }

            out << "\n";
        }

        auto t1_total = clock::now(); //τέλος συνολικής εκτέλεσης
        double total_sec = std::chrono::duration<double>(t1_total - t0_total).count(); //δευτερόλεπτα
        double QPS = queries.empty() ? 0.0 : queries.size() / total_sec; //queries per second

        out << "Average AF: " << (queries.empty() ? 0.0 : (sum_AF / queries.size())) << "\n";
        out << "Recall@N: " << (queries.empty() ? 0.0 : (sum_recall / queries.size())) << "\n";
        out << "QPS: " << QPS << "\n";
        out << "tApproximateAverage: " << (queries.empty() ? 0.0 : (sum_tApprox / queries.size())) << " ms\n";
        out << "tTrueAverage: " << tTrueAverage_ms << " ms\n";
    }

    /*nn*/
    template <typename IndexType>
    int nn_query(const IndexType &index, const std::vector<float> &q, float epsilon)
    {
        auto candidates = index.query_candidates(q); //λήψη υποψηφίων
        const auto &data = index.data(); //πρόσβαση στα δεδομένα
        auto dist_func = index.distance_func(); //συνάρτηση απόστασης

        if (candidates.empty()) 
            return -1;

        int best_id = -1; //αποθήκευση καλύτερου id
        float best_dist = std::numeric_limits<float>::max(); //αρχικοποίηση καλύτερης απόστασης
        /*ακριβής αναζήτηση μεταξύ των υποψηφίων*/
        for (int id : candidates)
        {
            float dist = dist_func(q, data[id]); //υπολογισμός απόστασης
            if (dist < best_dist)
            {
                best_dist = dist; //ενημέρωση καλύτερης απόστασης
                best_id = id; //ενημέρωση καλύτερου id
            }
        }

        /*ε-approximate check (προαιρετικό refinement)*/
        for (int i = 0; i < (int)data.size(); ++i)
        {
            float dist = dist_func(q, data[i]); //υπολογισμός απόστασης
            if (dist < best_dist / (1 + epsilon))
            {
                best_dist = dist; //ενημέρωση καλύτερης απόστασης
                best_id = i; //ενημέρωση καλύτερου id
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
        auto candidates = index.query_candidates(q); //λήψη υποψηφίων
        const auto &data = index.data(); //πρόσβαση στα δεδομένα
        auto dist_func = index.distance_func(); //συνάρτηση απόστασης
        /*υπολογισμός αποστάσεων*/
        std::vector<std::pair<float, int>> dists;
        dists.reserve(candidates.size()); //απόσταση, id
        for (int id : candidates)
            dists.emplace_back(dist_func(q, data[id]), id); //αποθήκευση (απόσταση, id)

        if ((int)dists.size() > N) //μερική ταξινόμηση για top-N
            std::nth_element(dists.begin(), dists.begin() + N, dists.end());
        else
            N = dists.size(); //προσαρμογή N αν λιγότεροι υποψήφιοι

        std::sort(dists.begin(), dists.begin() + N); //ταξινόμηση των top-N
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
        auto candidates = index.query_candidates(q); //λήψη υποψηφίων
        const auto &data = index.data(); //πρόσβαση στα δεδομένα
        auto dist_func = index.distance_func(); //συνάρτηση απόστασης
        /*υπολογισμός αποστάσεων*/
        std::vector<int> result;
        for (int id : candidates)
        {
            float dist = dist_func(q, data[id]); //υπολογισμός απόστασης
            if (dist <= R)
                result.push_back(id); //αποθήκευση id εντός ακτίνας R
        }
        return result;
    }

} /*namespace utils*/
