#include "ivfpq.h" //για την κλάση IVFPQ και ProductQuantizer
#include <algorithm> //για std::nth_element, std::sort
#include <iostream> //για std::cout, std::cerr
#include <cmath> //για std::sqrt
#include <unordered_set> //για std::unordered_set

namespace ivf
{

  //ProductQuantizer
  ProductQuantizer::ProductQuantizer(int M, int nbits, utils::DistanceFunc dist)
      : M_(M), nbits_(nbits), Ks_(1 << nbits), dist_func_(dist)
  {
    if (M_ <= 0)
      M_ = 1; //τουλάχιστον ένας υποχώρος
    if (nbits_ <= 0)
    {
      nbits_ = 8; //προεπιλογή 8 bits
      Ks_ = 256; //2^8
    }
  }

  //(επανα)διαχωρισμός ενός D-διαστατικού χώρου σε M μπλοκ (σχεδόν ίσα)
  void ProductQuantizer::set_dimension_splits(int D)
  {
    starts_.assign(M_, 0); //αρχικές διαστάσεις υποχώρων
    sizes_.assign(M_, 0); //μεγέθη υποχώρων

    int base = D / M_; //βάση μέγεθος υποχώρου
    int rem = D % M_; //υπόλοιπο για κατανομή
    int start = 0; //τρέχουσα αρχική διάσταση

    for (int m = 0; m < M_; ++m)
    {
      int sz = base + (m < rem ? 1 : 0); //κατανομή υπολοίπου
      starts_[m] = start; //αρχική διάσταση υποχώρου m
      sizes_[m] = sz; //μέγεθος υποχώρου m
      start += sz; //ενημέρωση αρχικής διάστασης για επόμενο υποχώρο
    }
  }

  //εκπαίδευση codebooks πάνω σε residuals
  void ProductQuantizer::train(const std::vector<std::vector<float>> &residuals)
  {
    if (residuals.empty()) //τίποτα για εκπαίδευση
      return;

    int D = static_cast<int>(residuals[0].size()); //διάσταση δεδομένων
    set_dimension_splits(D); //διαχωρισμός διαστάσεων σε M υποχώρους

    codebooks_.assign(M_, {}); //αρχικοποίηση codebooks
    std::mt19937 rng(1234567); //σταθερός σπόρος για αναπαραγωγιμότητα

    /* εκπαίδευση ενός KMeans σε κάθε υποχώρο ανεξάρτητα */
    for (int m = 0; m < M_; ++m)
    {
      int d_m = sizes_[m]; //διάσταση υποχώρου m
      int s_m = starts_[m]; //αρχική διάσταση υποχώρου m

      //υποσύνολο residuals για τον υποχώρο m
      std::vector<std::vector<float>> subtrain; //n x d_m
      subtrain.reserve(residuals.size()); //κράτηση χώρου
      for (const auto &r : residuals)
      {
        std::vector<float> sub(d_m); //υποχώρος m
        for (int j = 0; j < d_m; ++j)
          sub[j] = r[s_m + j]; //αντιγραφή διαστάσεων
        subtrain.emplace_back(std::move(sub)); //προσθήκη στο υποσύνολο
      }

      //εκπαίδευση KMeans
      clustering::KMeans kmeans(Ks_, 100, 1e-4f,
                                clustering::KMeans::InitMethod::KMEANS_PLUS_PLUS,
                                1337 + m, false, dist_func_);

      //αν δεν υπάρχουν αρκετά δείγματα για Ks κέντρα
      if ((int)subtrain.size() < Ks_)
      {
        int newKs = std::max(2, (int)subtrain.size()); //τουλάχιστον 2 κέντρα
        std::cerr << "[PQ] Reducing Ks in subspace " << m
                  << " to " << newKs << " (insufficient data)\n";
        clustering::KMeans small(newKs, 100, 1e-4f,
                                 clustering::KMeans::InitMethod::KMEANS_PLUS_PLUS,
                                 1337 + m, false, dist_func_); //μικρότερο KMeans
        small.fit(subtrain); //εκπαίδευση
        const auto &C = small.centroids(); //κέντρα από μικρότερο KMeans
        codebooks_[m].assign(Ks_, std::vector<float>(d_m, 0.0f)); //αρχικοποίηση codebook m
        for (int k = 0; k < Ks_; ++k)
          codebooks_[m][k] = C[k % newKs]; //αντιγραφή με επανάληψη
        continue;
      }

      kmeans.fit(subtrain); //εκπαίδευση KMeans
      codebooks_[m] = kmeans.centroids(); //αποθήκευση κέντρων ως codebook

      if ((int)codebooks_[m].size() < Ks_)
      {
        int have = codebooks_[m].size(); //πόσα κέντρα έχουμε
        codebooks_[m].resize(Ks_, std::vector<float>(d_m, 0.0f)); //αύξηση μεγέθους codebook
        for (int k = have; k < Ks_; ++k)
          codebooks_[m][k] = codebooks_[m][k % have]; //αντιγραφή με επανάληψη
      }
    }
  }

  //κωδικοποίηση ενός residual διανύσματος
  std::vector<uint16_t> ProductQuantizer::encode(const std::vector<float> &residual) const
  {
    std::vector<uint16_t> code(M_, 0); //κωδικός μήκους M
    for (int m = 0; m < M_; ++m)
    {
      int d_m = sizes_[m]; //διάσταση υποχώρου m
      int s_m = starts_[m]; //αρχική διάσταση υποχώρου m  

      float best = std::numeric_limits<float>::max(); //καλύτερη απόσταση
      int bestk = 0; //καλύτερο κέντρο

      for (int k = 0; k < Ks_; ++k)
      {
        const auto &c = codebooks_[m][k]; //κέντρο k υποχώρου m
        float dist = 0.0f; //απόσταση residual - κέντρο
        for (int j = 0; j < d_m; ++j)
        {
          float diff = residual[s_m + j] - c[j]; //διαφορά
          dist += diff * diff; //τετράγωνο διαφοράς
        } 
        if (dist < best)
        {
          best = dist; //ενημέρωση καλύτερης απόστασης
          bestk = k; //ενημέρωση καλύτερου κέντρου
        }
      }
      code[m] = static_cast<uint16_t>(bestk); //αποθήκευση δείκτη κέντρου στο κωδικό
    }
    return code;
  }

  //κατασκευή LUT για το residual
  std::vector<std::vector<float>> ProductQuantizer::compute_LUT(const std::vector<float> &residual_q) const
  {
    std::vector<std::vector<float>> lut(M_, std::vector<float>(Ks_, 0.0f)); //LUT M x Ks
    for (int m = 0; m < M_; ++m)
    {
      int d_m = sizes_[m]; //διάσταση υποχώρου m 
      int s_m = starts_[m]; //αρχική διάσταση υποχώρου m

      for (int k = 0; k < Ks_; ++k)
      {
        const auto &c = codebooks_[m][k]; //κέντρο k υποχώρου m
        float dist = 0.0f; //απόσταση residual_q - κέντρο
        for (int j = 0; j < d_m; ++j)
        {
          float diff = residual_q[s_m + j] - c[j]; //διαφορά
          dist += diff * diff; //τετράγωνο διαφοράς
        }
        lut[m][k] = dist; //αποθήκευση απόστασης στο LUT
      }
    }
    return lut;
  }

  //απόσταση ADC χρησιμοποιώντας LUT και code
  float ProductQuantizer::adc_distance(const std::vector<std::vector<float>> &lut,
                                       const std::vector<uint16_t> &code)
  {
    float d = 0.0f; //απόσταση ADC
    for (size_t m = 0; m < code.size(); ++m)
      d += lut[m][code[m]]; //προσθήκη απόστασης υποχώρου m
    return d;
  }

  /*IVFPQ*/
  IVFPQ::IVFPQ(int kclusters, int nprobe, int M, int nbits,
               unsigned int seed, int N, float R, utils::DistanceFunc dist_func)
      : kclusters_(kclusters),
        nprobe_(nprobe),
        M_(M),
        nbits_(nbits),
        Ks_(1 << nbits),
        seed_(seed),
        N_(N),
        R_(R),
        dist_func_(dist_func ? dist_func : utils::euclidean_distance),
        pq_(M, nbits, dist_func_)
  {
    std::cout << "IVFPQ initialized with kclusters=" << kclusters_
              << ", nprobe=" << nprobe_
              << ", M=" << M_
              << ", nbits=" << nbits_
              << ", seed=" << seed_
              << ", N=" << N_
              << ", R=" << R_ << std::endl;
  }

  //εύρεση κοντινότερου κεντροειδούς
  int IVFPQ::nearest_centroid(const std::vector<float> &x) const
  {
    float best = std::numeric_limits<float>::max(); //καλύτερη απόσταση
    int bestk = 0; //καλύτερο κέντρο
    for (int k = 0; k < kclusters_; ++k)
    {
      float d = dist_func_(x, centroids_[k]); //απόσταση από κέντρο k
      if (d < best)
      {
        best = d; //ενημέρωση καλύτερης απόστασης
        bestk = k; //ενημέρωση καλύτερου κέντρου
      }
    }
    return bestk; //επιστροφή δείκτη κοντινότερου κέντρου
  }

  //εύρεση nprobe κοντινότερων κεντροειδών
  std::vector<int> IVFPQ::top_nprobe_centroids(const std::vector<float> &q) const
  {
    std::vector<std::pair<float, int>> ds; //αποστάσεις και δείκτες κέντρων
    ds.reserve(kclusters_); //κράτηση χώρου
    for (int k = 0; k < kclusters_; ++k)
      ds.emplace_back(dist_func_(q, centroids_[k]), k); //υπολογισμός απόστασης

    int take = std::min(nprobe_, kclusters_); //πόσα να πάρουμε
    std::nth_element(ds.begin(), ds.begin() + take, ds.end()); //μερική ταξινόμηση
    ds.resize(take); //περιορισμός στα nprobe καλύτερα

    std::vector<int> ids; //αποθήκευση δεικτών κέντρων
    ids.reserve(take); //κράτηση χώρου
    for (auto &p : ds)
      ids.push_back(p.second); //προσθήκη δείκτη κέντρου
    return ids;
  }

  //υπολογισμός residual
  std::vector<float> IVFPQ::residual_of(const std::vector<float> &x,
                                        const std::vector<float> &c)
  {
    std::vector<float> r(x.size()); //αποθήκευση residual
    for (size_t i = 0; i < x.size(); ++i)
      r[i] = x[i] - c[i]; //υπολογισμός residual
    return r;
  }

  //κατασκευή ευρετηρίου
  void IVFPQ::build_index(const std::vector<std::vector<float>> &data)
  {
    if (data.empty()) //τίποτα για ευρετήριο
      return;

    data_ptr_ = &data; //αποθήκευση δείκτη στα δεδομένα
    size_t n = data.size(); //αριθμός δεδομένων

    /* 1) επιλογή k μέσω silhouette αν χρειάζεται */
    int k_opt = kclusters_; //αρχική τιμή k
    if (k_opt <= 0) //αυτόματη επιλογή k
    {
      int k_min = 2; //ελάχιστο k
      int k_max = std::min<int>(10, std::sqrt(n)); //μέγιστο k
      using namespace clustering; //για KMeans και Silhouette
      float best_score = -1.0f; //καλύτερο silhouette score
      int best_k = k_min; //καλύτερο k
      for (int k = k_min; k <= k_max; ++k)
      {
        KMeans kmeans(k, 100, 1e-4f, KMeans::InitMethod::KMEANS_PLUS_PLUS, seed_, false, dist_func_); //KMeans
        kmeans.fit(data); //εκπαίδευση
        Silhouette sil(dist_func_); //Silhouette
        float s = sil.compute(data, kmeans.labels(), k); //υπολογισμός silhouette score
        if (s > best_score)
        {
          best_score = s; //ενημέρωση καλύτερου score
          best_k = k; //ενημέρωση καλύτερου k
        }
      }
      k_opt = best_k; //επιλογή καλύτερου k
      std::cerr << "[IVFPQ] Selected k=" << k_opt << " via silhouette.\n";
    }
    kclusters_ = k_opt; //ορισμός τελικού k

    /* 2) K-Means σε υποσύνολο για coarse quantizer */
    size_t subset_size = std::max<size_t>(kclusters_, (size_t)std::sqrt(n)); //μέγεθος υποσυνόλου
    std::mt19937 rng(seed_); //RNG με δοσμένο σπόρο
    std::uniform_int_distribution<size_t> rnd(0, n - 1); //κατανομή δεικτών
    std::unordered_set<size_t> pick; //για αποφυγή διπλοεπιλογών
    std::vector<std::vector<float>> subset; //αποθηκευτικό υποσυνόλου
    subset.reserve(subset_size); //κράτηση χώρου

    while (subset.size() < subset_size)
    {
      size_t i = rnd(rng); //τυχαίος δείκτης
      if (pick.insert(i).second)
        subset.push_back(data[i]); //προσθήκη στο υποσύνολο
    }

    //εκπαίδευση KMeans
    clustering::KMeans km(kclusters_, 100, 1e-4f,
                          clustering::KMeans::InitMethod::KMEANS_PLUS_PLUS,
                          seed_, false, dist_func_);
    km.fit(subset); //εκπαίδευση
    centroids_ = km.centroids(); //αποθήκευση κεντροειδών

    /* 3) συλλογή residuals για εκπαίδευση του PQ */
    invlists_.assign(kclusters_, {}); //αρχικοποίηση κενών λιστών
    std::vector<std::vector<float>> residuals_for_pq; //συλλογή residuals
    residuals_for_pq.reserve(subset_size); //κράτηση χώρου

    for (size_t i = 0; i < n; ++i)
    {
      int cid = nearest_centroid(data[i]); //κοντινότερο κέντρο
      auto r = residual_of(data[i], centroids_[cid]); //υπολογισμός residual
      residuals_for_pq.emplace_back(std::move(r)); //προσθήκη residual
    }

    if ((int)residuals_for_pq.size() > 20000)
    {
      std::vector<std::vector<float>> small; //μικρότερο σύνολο για PQ
      small.reserve(20000); //κράτηση χώρου
      std::unordered_set<size_t> seen; //για αποφυγή διπλοεπιλογών
      while (small.size() < 20000)
      {
        size_t idx = rnd(rng) % residuals_for_pq.size(); //τυχαίος δείκτης
        if (seen.insert(idx).second)
          small.push_back(residuals_for_pq[idx]); //προσθήκη στο μικρό σύνολο
      }
      residuals_for_pq.swap(small); //αντικατάσταση με μικρότερο σύνολο
    }

    /* 4) εκπαίδευση του PQ πάνω στα residuals */
    pq_.train(residuals_for_pq);

    /* 5) κωδικοποίηση residuals και κατασκευή inverted lists */
    for (size_t i = 0; i < n; ++i)
    {
      int cid = nearest_centroid(data[i]); //κοντινότερο κέντρο
      auto r = residual_of(data[i], centroids_[cid]); //υπολογισμός residual
      auto code = pq_.encode(r); //κωδικοποίηση residual
      invlists_[cid].push_back({static_cast<int>(i), std::move(code)}); //προσθήκη στη λίστα
    }

    std::cerr << "[IVFPQ] Built index with " << kclusters_
              << " lists; PQ(M=" << M_ << ", nbits=" << nbits_ << ").\n";
  }

  //λήψη ταυτοτήτων υποψηφίων
  std::vector<int> IVFPQ::query_candidates(const std::vector<float> &q) const
  {
    std::vector<int> ids; //αποθήκευση ταυτοτήτων υποψηφίων
    auto topc = top_nprobe_centroids(q); //εύρεση nprobe κοντινότερων κεντροειδών
    for (int cid : topc)
      for (const auto &e : invlists_[cid])
        ids.push_back(e.id); //προσθήκη ταυτότητας υποψηφίου
    return ids;
  }

  //αναζήτηση KNN χρησιμοποιώντας ADC
  std::vector<std::pair<int, float>> IVFPQ::knn_query(const std::vector<float> &q, int N) const
  {
    std::vector<std::pair<int, float>> out; //αποθήκευση αποτελεσμάτων
    if (!data_ptr_ || centroids_.empty())
      return out;

    struct Pair { float dist; int id; }; //απόσταση και ταυτότητα
    std::vector<Pair> heap; //αποθήκευση υποψηφίων

    auto topc = top_nprobe_centroids(q); //εύρεση nprobe κοντινότερων κεντροειδών
    for (int cid : topc)
    {
      auto rq = residual_of(q, centroids_[cid]); //υπολογισμός residual ερωτήματος
      auto lut = pq_.compute_LUT(rq); //κατασκευή LUT
      for (const auto &e : invlists_[cid])
      {
        float d = ProductQuantizer::adc_distance(lut, e.code); //απόσταση ADC
        heap.push_back({d, e.id}); //προσθήκη υποψηφίου
      }
    }

    if (heap.empty()) //κανένας υποψήφιος
      return out;

    if ((int)heap.size() > N) //περισσότεροι από N υποψήφιοι
    {
      std::nth_element(heap.begin(), heap.begin() + N, heap.end(),
                       [](const Pair &a, const Pair &b) { return a.dist < b.dist; }); //μερική ταξινόμηση
      heap.resize(N); //περιορισμός σε N καλύτερους
    } 
    std::sort(heap.begin(), heap.end(),
              [](const Pair &a, const Pair &b) { return a.dist < b.dist; }); //ταξινόμηση των N καλύτερων

    out.reserve(heap.size()); //κράτηση χώρου
    for (auto &p : heap)
      out.emplace_back(p.id, p.dist); //μετατροπή σε ζεύγη (id, απόσταση)
    return out;
  }

  //αναζήτηση περιοχής (range) χρησιμοποιώντας ADC
  std::vector<int> IVFPQ::range_search(const std::vector<float> &q, float R) const
  {
    std::vector<int> ids; //αποθήκευση ταυτοτήτων εντός ακτίνας
    if (!data_ptr_)
      return ids;

    auto topc = top_nprobe_centroids(q); //εύρεση nprobe κοντινότερων κεντροειδών
    for (int cid : topc)
    {
      auto rq = residual_of(q, centroids_[cid]); //υπολογισμός residual ερωτήματος
      auto lut = pq_.compute_LUT(rq); //κατασκευή LUT
      for (const auto &e : invlists_[cid])
      {
        float d = ProductQuantizer::adc_distance(lut, e.code); //απόσταση ADC
        if (d <= R)
          ids.push_back(e.id); //προσθήκη ταυτότητας εντός ακτίνας
      }
    }
    return ids;
  }

  //διαγραφή ευρετηρίου
  void IVFPQ::clear_index()
  {
    centroids_.clear(); //καθαρισμός κεντροειδών
    invlists_.clear(); //καθαρισμός λιστών
    data_ptr_ = nullptr; //αφαίρεση δείκτη δεδομένων
  }

} /*namespace ivf*/
