import pandas as pd #εισαγωγή της βιβλιοθήκης pandas

#λίστα με τα αρχεία
data_files = ["silhouette_results_sift.txt", "silhouette_results_mnist.txt"]

#δημιουργία κενής λίστας για τα δεδομένα
all_data = []

#διαβάζουμε και τα δύο αρχεία
for data_file in data_files:
    #διαβάζουμε το όνομα της κατηγορίας (sift ή mnist) από το όνομα του αρχείου
    type = data_file.split("_")[-1].split(".")[0]  #εξάγουμε το "sift" ή "mnist" από το όνομα του αρχείου
    
    #άνοιγμα και ανάγνωση του .txt αρχείου
    with open(data_file, "r") as file:
        lines = file.readlines()
    
    #διαβάζουμε και αποθηκεύουμε τα δεδομένα για κάθε αρχείο
    for line in lines[1:]:  #αγνοούμε την πρώτη γραμμή, αν είναι ο τίτλος
        k, score = line.strip().split()  #χωρίζουμε το k και το score
        all_data.append([int(k), float(score), type])  #προσθέτουμε και την κατηγορία

#δημιουργία DataFrame
df = pd.DataFrame(all_data, columns=["k", "silhouette", "type"])

#αποθήκευση στο αρχείο .xlsx χρησιμοποιώντας pandas
df.to_excel("silhouette_results_combined.xlsx", index=False)
print("Results saved to silhouette_results_combined.xlsx")
