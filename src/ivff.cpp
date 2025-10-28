#include "ivff.h" //η κεφαλίδα της κλάσης IVFFlat
#include <algorithm> //std::sort, std::nth_element
#include <iostream> //std::cout, std::cerr
#include <cmath> //std::sqrt

namespace ivf {

//κονστράκτορας 
IVFFlat::IVFFlat(int kclusters, int nprobe, unsigned int seed, int N, float R, utils::DistanceFunc dist_func)
    : kclusters_(kclusters),
      nprobe_(nprobe),
      seed_(seed),
      N_(N),
      R_(R),
      dist_func_(dist_func ? dist_func : utils::euclidean_distance) //αν δεν δοθεί, default
{
    std::cout << "IVFFlat initialized with kclusters=" << kclusters_ 
              << ", nprobe=" << nprobe_
              << ", seed=" << seed_
              << ", N=" << N_
              << ", R=" << R_ << std::endl;
}

//κατασκευή του index
void IVFFlat::build_index(const std::vector<std::vector<float>>& data) {
    if (data.empty()) return;
    data_ptr_ = &data; //αποθήκευση pointer στα δεδομένα
    std::size_t n = data.size(); //αριθμός σημείων

    //1) Επιλογή του αριθμού συστάδων k με silhouette (αν nlist_ <= 0)
    int k_opt = kclusters_; //αρχικά το k από τον χρήστη
    if (k_opt <= 0) { //πρέπει να το επιλέξουμε
        int k_min = 2; //τουλάχιστον 2 clusters
        int k_max = std::min<int>(10, std::sqrt(n)); //μην το παρακάνουμε
        std::cerr << "[IVF] Selecting best k via Silhouette in range ["
                  << k_min << "," << k_max << "]...\n";

        using namespace clustering;
        float best_score = -1.0f; //αρχικά πολύ χαμηλό
        int best_k = k_min; //αρχικά το μικρότερο

        for (int k = k_min; k <= k_max; ++k) {
            KMeans kmeans(k, 100, 1e-4f,
                          KMeans::InitMethod::KMEANS_PLUS_PLUS,
                          seed_, false, dist_func_); //αρχικοποίηση KMeans
            kmeans.fit(data); //εκπαίδευση

            Silhouette sil(dist_func_); //υπολογισμός silhouette
            float score = sil.compute(data, kmeans.labels(), k); //υπολογισμός score

            std::cerr << "[Silhouette] k=" << k
                      << " score=" << score << std::endl;

            if (score > best_score) {
                best_score = score; //ενημέρωση καλύτερου score
                best_k = k; //ενημέρωση καλύτερου k
            }
        }

        k_opt = best_k; //επιλογή του καλύτερου k
        std::cerr << "[Silhouette] Best k=" << k_opt
                  << " (score=" << best_score << ")\n";
    }

    //2) Επιλογή υποσυνόλου X'
    std::size_t subset_size = static_cast<std::size_t>(std::sqrt(n));
    if (subset_size < static_cast<std::size_t>(k_opt))
        subset_size = k_opt; //τουλάχιστον k σημεία

    std::vector<std::vector<float>> subset; //υποσύνολο δεδομένων
    subset.reserve(subset_size); //κράτημα χώρου

    std::mt19937 rng(seed_); //RNG με σπόρο
    std::uniform_int_distribution<std::size_t> dist_idx(0, n - 1); //κατανομή δεικτών
    std::unordered_set<std::size_t> picked; //για αποφυγή διπλοεγγραφών

    while (subset.size() < subset_size) {
        std::size_t idx = dist_idx(rng); //τυχαίος δείκτης
        if (picked.insert(idx).second)
            subset.push_back(data[idx]); //προσθήκη αν δεν έχει επιλεγεί ήδη
    }

    //3) Εκτέλεση K-Means με το βέλτιστο k
    clustering::KMeans kmeans(k_opt, 100, 1e-4f,
                              clustering::KMeans::InitMethod::KMEANS_PLUS_PLUS,
                              seed_, false, dist_func_);
    kmeans.fit(subset); //εκπαίδευση
    centroids_ = kmeans.centroids(); //αποθήκευση centroids
    kclusters_ = k_opt; //ενημέρωση του αριθμού clusters

    //4) Ανάθεση σημείων στο κοντινότερο centroid
    inverted_lists_.assign(kclusters_, {}); //αρχικοποίηση inverted lists
    for (std::size_t i = 0; i < n; ++i) {
        const auto& x = data[i]; //τρέχον σημείο
        float bestd = std::numeric_limits<float>::max(); //αρχικά πολύ μεγάλο
        int bestk = -1; //κανένα cluster
        for (int k = 0; k < kclusters_; ++k) {
            float d = dist_func_(x, centroids_[k]); //απόσταση από το centroid
            if (d < bestd) {
                bestd = d; //ενημέρωση καλύτερης απόστασης
                bestk = k; //ενημέρωση καλύτερου cluster
            }
        }
        inverted_lists_[bestk].push_back(static_cast<int>(i)); //προσθήκη στο inverted list
    }

    std::cerr << "[IVF] Built index with " << kclusters_
              << " clusters (chosen by silhouette)." << std::endl;
}

//εύρεση υποψηφίων (βήμα coarse search)
std::vector<int> IVFFlat::query_candidates(const std::vector<float>& q) const {
    std::vector<std::pair<float,int>> centroid_dists; //αποστάσεις από centroids
    centroid_dists.reserve(kclusters_); //κράτημα χώρου
    for (int k = 0; k < kclusters_; ++k) {
        float d = dist_func_(q, centroids_[k]); //απόσταση από το centroid
        centroid_dists.emplace_back(d, k); //αποθήκευση απόστασης και id centroid
    }

    //κρατάμε τα nprobe κοντινότερα centroids
    std::nth_element(centroid_dists.begin(),
                     centroid_dists.begin() + std::min(nprobe_, kclusters_),
                     centroid_dists.end()); //μερική ταξινόμηση
    centroid_dists.resize(std::min(nprobe_, kclusters_)); //περιορισμός στο nprobe

    //ενώνουμε τα inverted lists από τα πιο κοντινά centroids
    std::vector<int> candidates;
    for (auto& [_, cid] : centroid_dists)
        for (int idx : inverted_lists_[cid])
            candidates.push_back(idx); //προσθήκη υποψηφίων

    return candidates;
}

//αναζήτηση Ν κοντινότερων γειτόνων
std::vector<std::pair<int,float>> IVFFlat::knn_query(const std::vector<float>& q, int N) const {
    std::vector<std::pair<int,float>> results; //αποτέλεσμα
    if (!data_ptr_ || centroids_.empty()) return results;

    auto candidates = query_candidates(q); //λήψη υποψηφίων
    if (candidates.empty()) return results; //κανένας υποψήφιος

    std::vector<std::pair<float,int>> dists; //αποστάσεις υποψηφίων
    dists.reserve(candidates.size()); //κράτημα χώρου
    for (int idx : candidates) {
        float d = dist_func_(q, (*data_ptr_)[idx]); //απόσταση
        dists.emplace_back(d, idx); //αποθήκευση απόστασης και id
    }

    if (dists.size() > static_cast<std::size_t>(N))
        std::nth_element(dists.begin(), dists.begin() + N, dists.end()); //μερική ταξινόμηση
    else
        N = static_cast<int>(dists.size()); //προσαρμογή N αν λιγότεροι υποψήφιοι

    std::sort(dists.begin(), dists.begin() + N); //ταξινόμηση των N κοντινότερων

    for (int i = 0; i < N; ++i)
        results.emplace_back(dists[i].second, dists[i].first); //αποθήκευση στο αποτέλεσμα

    return results;
}

//range search
std::vector<int> IVFFlat::range_search(const std::vector<float>& q, float R) const {
    std::vector<int> result; //αποτέλεσμα
    if (!data_ptr_) return result;

    auto candidates = query_candidates(q); //λήψη υποψηφίων
    for (int idx : candidates) {
        float d = dist_func_(q, (*data_ptr_)[idx]); //απόσταση
        if (d <= R)
            result.push_back(idx); //προσθήκη στο αποτέλεσμα
    }
    return result;
}

//καθαρισμός
void IVFFlat::clear_index() {
    centroids_.clear(); //καθαρισμός centroids
    inverted_lists_.clear(); //καθαρισμός inverted lists
    data_ptr_ = nullptr; //αφαίρεση pointer στα δεδομένα
    std::cerr << "[IVF] Index cleared.\n";
}

} //namespace ivf
