#include "hypercube.h" //για την κλάση Hypercube
#include <algorithm> //για std::partial_sort
#include <random> //για std::mt19937, std::normal_distribution, std::uniform_real_distribution
#include <unordered_set> //για std::unordered_set
#include <queue> //για std::queue
#include <iostream> //για debugging

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
        uint64_t vertex = 0; //αρχικοποίηση του vertex id σε 0

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
            uint8_t bit; //bit {0,1} για την τρέχουσα προβολή
            if (it != bit_map_[i].end())
            {
                bit = it->second; //χρήση αποθηκευμένου bit
            }
            else
            {
                /*δημιουργία ντετερμινιστικού bit {0,1} με χρήση hash + seed ώστε να είναι επαναλήψιμο*/
                std::mt19937 gen(seed_ ^ (i * 1315423911u) ^ (std::hash<long long>{}(h)));
                bit = gen() % 2; //τυχαίο bit {0,1}
                bit_map_[i][h] = bit; //αποθήκευση στην bit_map_
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
        data_ptr_ = &data; //αποθήκευση δείκτη στα δεδομένα

        std::mt19937 gen(seed_); //αρχικοποίηση RNG με δεδομένο seed
        std::normal_distribution<float> normal(0.0f, 1.0f); //κανονική κατανομή N(0,1)
        std::uniform_real_distribution<float> uniform(0.0f, w_); //ομοιόμορφη κατανομή [0, w)

        /*δημιουργία τυχαίων διανυσμάτων προβολής και μετατοπίσεων*/
        v_.assign(dprime_, std::vector<float>(dim_)); //dprime_ διανύσματα διαστάσεων dim_
        t_.assign(dprime_, 0.0f); //dprime_ μετατοπίσεις
        bit_map_.assign(dprime_, {}); //καθαρισμός bit_map_

        for (int i = 0; i < dprime_; ++i)
        {
            for (int j = 0; j < dim_; ++j)
                v_[i][j] = normal(gen); //τυχαίο Gaussian στοιχείο
            t_[i] = uniform(gen); //τυχαία μετατόπιση
        }

        /*καθαρισμός προηγούμενων κάδων*/
        vertex_buckets_.clear();

        /*αντιστοίχιση κάθε σημείου σε κορυφή*/
        for (int idx = 0; idx < (int)data.size(); ++idx)
        {
            uint64_t vertex = point_to_vertex(data[idx]); //υπολογισμός vertex id
            vertex_buckets_[vertex].push_back(idx); //αποθήκευση index στον κατάλληλο κάδο
        }
    }

    /*αφαίρεση όλων των δεδομένων*/
    void Hypercube::clear_index()
    {
        vertex_buckets_.clear();  //καθαρισμός κάδων
        bit_map_.clear(); //καθαρισμός bit_map_
        v_.clear(); //καθαρισμός διανυσμάτων προβολής
        t_.clear(); //καθαρισμός μετατοπίσεων
    }

    /*Συλλογή υποψήφιων σημείων από κοντινές κορυφές του hypercube.
      Διασχίζει κορυφές με BFS, αυξάνοντας προοδευτικά την απόσταση Hamming.*/
    void Hypercube::collect_neighbors_by_hamming(uint64_t vertex, std::vector<int> &out_candidates) const
    {
        std::queue<uint64_t> q;  //ουρά για BFS
        std::unordered_set<uint64_t> visited; //σύνολο επισκεφθέντων κορυφών
        q.push(vertex); //ξεκινά από την αρχική κορυφή
        visited.insert(vertex); //σημειώνει ως επισκεφθέν

        int probes = 0; //μετρητής για το μέγιστο probes
        while (!q.empty() && probes < max_probes_)
        {
            uint64_t v = q.front(); //τρέχουσα κορυφή
            q.pop(); //αφαίρεση από την ουρά
            ++probes; //αύξηση μετρητή probes

            /*αν υπάρχει bucket για αυτήν την κορυφή, προσθέτει τα indices*/
            auto it = vertex_buckets_.find(v); //αναζήτηση κάδου
            if (it != vertex_buckets_.end()) //βρέθηκε κάδος
            {
                for (int idx : it->second)
                    out_candidates.push_back(idx); //προσθήκη index στους υποψηφίους
                if ((int)out_candidates.size() >= max_candidates_) //έλεγχος μέγιστου αριθμού υποψηφίων
                    return;
            }

            /*δημιουργία γειτόνων με Hamming απόσταση 1*/
            for (int b = 0; b < dprime_; ++b)
            {
                uint64_t neighbor = v ^ (1ULL << b); //αναστροφή του b-οστού bit
                if (!visited.count(neighbor))
                {
                    visited.insert(neighbor); //σημειώνει ως επισκεφθέν
                    q.push(neighbor); //προσθήκη στην ουρά για περαιτέρω εξερεύνηση
                }
            }
        }
    }

    /*επιστρέφει όλους τους υποψηφίους γείτονες για ένα ερώτημα q*/
    std::vector<int> Hypercube::query_candidates(const std::vector<float> &q) const
    {
        uint64_t vertex = point_to_vertex(q); //υπολογισμός vertex id για το ερώτημα
        std::vector<int> candidates; //αποθήκευση υποψηφίων γειτόνων
        collect_neighbors_by_hamming(vertex, candidates); //συλλογή υποψηφίων
        return candidates;
    }

    /*Εκτέλεση ερωτήματος k-NN:
      - Παίρνει όλους τους υποψηφίους (από κοντινές κορυφές)
      - Υπολογίζει αποστάσεις
      - Κρατά τους Ν μικρότερους*/
    std::vector<std::pair<int, float>> Hypercube::knn_query(const std::vector<float> &q, int N) const
    {
        std::vector<int> candidates = query_candidates(q); //λήψη υποψηφίων γειτόνων
        std::vector<std::pair<int, float>> results; //αποθήκευση αποτελεσμάτων (index, απόσταση)

        for (int idx : candidates)
        {
            float dist = dist_func_(q, (*data_ptr_)[idx]); //υπολογισμός απόστασης
            results.emplace_back(idx, dist); //αποθήκευση αποτελέσματος
        }

        /*ταξινόμηση(μερική) με βάση την απόσταση*/
        std::partial_sort(results.begin(), results.begin() + std::min(N, (int)results.size()),
                          results.end(), [](auto &a, auto &b) 
                          { return a.second < b.second; }); //συγκριτής με βάση την απόσταση

        if ((int)results.size() > N)
            results.resize(N); //περιορισμός στα N καλύτερα
        return results;
    }

    /*range search: επιστρέφει όλα τα σημεία με απόσταση ≤ R*/
    std::vector<int> Hypercube::range_search(const std::vector<float> &q, float R) const
    {
        std::vector<int> candidates = query_candidates(q); //λήψη υποψηφίων γειτόνων
        std::vector<int> in_range; //αποθήκευση σημείων εντός ακτίνας R
        for (int idx : candidates)
        {
            float dist = dist_func_(q, (*data_ptr_)[idx]);
            if (dist <= R)
                in_range.push_back(idx); //προσθήκη αν είναι εντός ακτίνας
        }
        return in_range;
    }

} /*namespace hcube*/
