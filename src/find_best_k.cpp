#include "mnist.h" //για τη φόρτωση των εικόνων MNIST
#include "sift.h" //για τη φόρτωση των SIFT descriptors
#include "kmeans.h" //για το KMeans
#include "silhouette.h" //για τον υπολογισμό του silhouette score
#include <iostream> //για είσοδο/έξοδο
#include <fstream> //για αρχεία
#include <string> //για συμβολοσειρές
#include <filesystem>  //για να δουλέψουμε με τα paths

namespace fs = std::filesystem; //σύντομο όνομα για το filesystem

using clustering::KMeans; //χρήση της κλάσης KMeans από το namespace clustering
using clustering::Silhouette; //χρήση της κλάσης Silhouette από το namespace clustering

int main(int argc, char** argv) { //κύρια συνάρτηση
    if (argc < 3) { //έλεγχος ορθότητας παραμέτρων
        std::cerr << "Usage: " << argv[0] 
                  << " -d <data_file> [max_images]" << std::endl; 
        return 1;
    }

    std::string data_file; //διαδρομή προς το αρχείο δεδομένων
    int max_images = 10000; //προεπιλεγμένος μέγιστος αριθμός εικόνων/διανυσμάτων

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i]; //τρέχουσα παράμετρος
        if (arg == "-d" && i + 1 < argc)
            data_file = argv[++i]; //αποθήκευση διαδρομής αρχείου
        else if (arg == "-n" && i + 1 < argc)
            max_images = std::stoi(argv[++i]); //αποθήκευση μέγιστου αριθμού εικόνων/διανυσμάτων
    }

    //ανίχνευση του dataset από τη διαδρομή του αρχείου
    std::string dataset = "Unknown";
    if (data_file.find("mnist") != std::string::npos) {
        dataset = "mnist"; //ανιχνεύτηκε MNIST
    } else if (data_file.find("sift") != std::string::npos) {
        dataset = "sift"; //ανιχνεύτηκε SIFT
    }

    if (dataset == "Unknown") { //αν δεν αναγνωρίστηκε το dataset
        std::cerr << "Unable to determine dataset type from the file path." << std::endl;
        return 1;
    }

    std::vector<std::vector<float>> data; //δομή για τα δεδομένα

    if (dataset == "mnist") {
        data = load_mnist_images<false>(data_file, max_images); //φόρτωση MNIST χωρίς κανονικοποίηση
    } else if (dataset == "sift") {
        data = sift::load_sift_dat(data_file, max_images, 128); //φόρτωση SIFT με αναμενόμενη διάσταση 128
    }

    std::cout << "Loaded " << data.size() << " samples.\n";

    //τιμές του k για τις οποίες θα υπολογίσουμε το silhouette score
    std::vector<int> k_values = {2, 3, 4, 5, 6, 7, 8, 9, 10};
    Silhouette silhouette;

    //δημιουργία του ονόματος του αρχείου output βασισμένο στο dataset
    std::string output_filename = (dataset == "mnist") ? "silhouette_results_mnist.txt" : "silhouette_results_sift.txt";
    std::ofstream fout(output_filename); //άνοιγμα του αρχείου για εγγραφή
    fout << "k silhouette\n";  //στήλες για το αρχείο .txt

    std::cout << "\nComputing silhouette scores for " << dataset << " dataset:\n";
    std::cout << "--------------------------------\n"; 

    //υπολογισμός του silhouette score για κάθε k
    for (int k : k_values) {
        KMeans kmeans(k, 100, 1e-4, KMeans::InitMethod::KMEANS_PLUS_PLUS, 42, false);
        kmeans.fit(data); //εκπαίδευση του KMeans
        float score = silhouette.compute(data, kmeans.labels(), k);

        //αποθήκευση αποτελέσματος αμέσως στο αρχείο .txt
        fout << k << " " << score << "\n";
        std::cout << "k = " << k << " → silhouette = " << score << std::endl;
    }

    fout.close(); //κλείσιμο του αρχείου
    std::cout << "\nResults saved in " << output_filename << std::endl;
    return 0;
}
