import os #για διαχείριση φακέλων και αρχείων
import re #για regex
import pandas as pd #για πίνακες δεδομένων
import matplotlib.pyplot as plt #για γραφήματα
import seaborn as sns #για βελτιωμένα γραφήματα
from matplotlib.backends.backend_pdf import PdfPages #για αποθήκευση πολλαπλών γραφημάτων σε PDF

#ρυθμίσεις φακέλων
RESULTS_DIR = "." #φάκελος με τα αρχεία αποτελεσμάτων
PLOTS_DIR = "plots" #φάκελος για αποθήκευση γραφημάτων και πίνακα
os.makedirs(PLOTS_DIR, exist_ok=True) #δημιουργία φακέλου αν δεν υπάρχει

#Regex patterns για τις μετρικές
metrics_pattern = {
    "Average AF": r"Average AF:\s*([0-9.eE+-]+)",
    "Recall@N": r"Recall@N:\s*([0-9.eE+-]+)",
    "QPS": r"QPS:\s*([0-9.eE+-]+)",
    "tApproximateAverage": r"tApproximateAverage:\s*([0-9.eE+-]+)",
    "tTrueAverage": r"tTrueAverage:\s*([0-9.eE+-]+)"
}

#συνάρτηση ανάγνωσης αποτελεσμάτων από αρχείο
def parse_results_file(filepath):
    with open(filepath, "r") as f:
        text = f.read() #ανάγνωση περιεχομένου αρχείου

    data = {} #αποθήκευση αποτελεσμάτων
    for key, pattern in metrics_pattern.items():
        match = re.search(pattern, text) #αναζήτηση μετρικής με regex
        data[key] = float(match.group(1)) if match else None #αποθήκευση τιμής ή None

    filename = os.path.basename(filepath).lower() #όνομα αρχείου σε πεζά

    #εντοπισμός μεθόδου
    for method in ["lsh", "hypercube", "ivfflat", "ivfpq"]:
        if method in filename:
            data["Method"] = method.upper() #αποθήκευση μεθόδου
            break
    else:
        data["Method"] = "Unknown" #αν δεν βρεθεί μέθοδος

    #εντοπισμός dataset
    for dataset in ["mnist", "sift"]:
        if dataset in filename:
            data["Dataset"] = dataset.upper() #αποθήκευση dataset
            break
    else:
        data["Dataset"] = "Unknown" #αν δεν βρεθεί dataset

    #Range Search: αν υπάρχει "range true" στο όνομα 
    if re.search(r"_true", filename):
        data["Range Search"] = "True"
    elif re.search(r"_false", filename):
        data["Range Search"] = "False"
    else:
        data["Range Search"] = "Unknown"  #προαιρετικά default τιμή

    return data


#ανάγνωση όλων των αρχείων αποτελεσμάτων
records = []
for fname in os.listdir(RESULTS_DIR):
    if fname.startswith("results_") and fname.endswith(".txt"):
        print(f"Reading: {fname}") #εκτύπωση ονόματος αρχείου
        filepath = os.path.join(RESULTS_DIR, fname) #πλήρης διαδρομή αρχείου
        records.append(parse_results_file(filepath)) #προσθήκη αποτελεσμάτων στη λίστα

if not records:
    print(" Δεν βρέθηκαν αρχεία results_*.txt στον φάκελο.")
    exit(1)

#δημιουργία DataFrame
df = pd.DataFrame(records)
df = df[["Method", "Dataset", "Range Search", "Average AF", "Recall@N", "QPS", "tApproximateAverage", "tTrueAverage"]]

print("\n Πίνακας συνολικών αποτελεσμάτων:\n") #εκτύπωση πίνακα
print(df.to_string(index=False)) #εκτύπωση χωρίς δείκτες

# Αποθήκευση πίνακα
table_path_csv = os.path.join(PLOTS_DIR, "results_table.csv") #αποθήκευση ως CSV
table_path_xlsx = os.path.join(PLOTS_DIR, "results_table.xlsx") #αποθήκευση ως Excel
df.to_csv(table_path_csv, index=False) #αποθήκευση CSV
df.to_excel(table_path_xlsx, index=False) #αποθήκευση Excel

print(f"\n Ο πίνακας αποθηκεύτηκε σε:\n- {table_path_csv}\n- {table_path_xlsx}")

#κρατάμε μόνο τις αριθμητικές στήλες για averaging
numeric_cols = ["Average AF", "Recall@N", "QPS", "tApproximateAverage", "tTrueAverage"]
df_grouped = df.groupby(["Dataset", "Method"], as_index=False)[numeric_cols].mean()

#pυθμίσεις εμφάνισης
sns.set(style="whitegrid", font_scale=1.15)

def plot_metric(metric, title, ylabel): #συνάρτηση για δημιουργία γραφήματος μετρικής
    plt.figure(figsize=(7, 5)) #δημιουργία νέας φιγούρας
    ax = sns.barplot(data=df_grouped, x="Dataset", y=metric, hue="Method", palette="Set2") #barplot με seaborn
    plt.title(title) #τίτλος γραφήματος
    plt.ylabel(ylabel) #ετικέτα άξονα y
    plt.xlabel("Dataset") #ετικέτα άξονα x
    plt.legend(title="Method", loc="best") #υπόμνημα

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9) #ετικέτες πάνω από τις μπάρες

    plt.tight_layout() #βελτιστοποίηση διάταξης
    safe_name = metric.replace("@", "_at_").replace(" ", "_") #ασφαλές όνομα αρχείου
    plt.savefig(f"{PLOTS_DIR}/{safe_name}.png", dpi=150) #αποθήκευση γραφήματος
    return plt.gcf()

# Δημιουργία όλων των διαγραμμάτων
plots = []
plots.append(plot_metric("Average AF", "Average Approximation Factor", "AF"))
plots.append(plot_metric("Recall@N", "Recall@N Comparison", "Recall"))
plots.append(plot_metric("QPS", "Queries Per Second", "QPS"))
plots.append(plot_metric("tApproximateAverage", "Average Approximate Query Time (ms)", "ms"))
plots.append(plot_metric("tTrueAverage", "Average True NN Time (ms)", "ms"))

# Scatter διάγραμμα ταχύτητας/ποιότητας
plt.figure(figsize=(7,5)) #δημιουργία νέας φιγούρας
sns.scatterplot( 
    data=df_grouped, x="QPS", y="Average AF",
    hue="Method", style="Dataset", s=120, palette="Set1"
) #scatter plot με seaborn
plt.title("Trade-off: Speed vs Approximation Quality") #τίτλος γραφήματος
plt.xlabel("Queries Per Second (QPS)") #ετικέτα άξονα x
plt.tight_layout() #βελτιστοποίηση διάταξης
plt.savefig(f"{PLOTS_DIR}/tradeoff_speed_vs_quality.png", dpi=150) #αποθήκευση γραφήματος
plots.append(plt.gcf()) #προσθήκη στη λίστα γραφημάτων

# Εξαγωγή όλων σε PDF
pdf_path = os.path.join(PLOTS_DIR, "summary_report.pdf")
with PdfPages(pdf_path) as pdf:
    for fig in plots:
        pdf.savefig(fig) #αποθήκευση κάθε γραφήματος σε PDF

print(f"\n Summary report saved: {pdf_path}")
print(f"\n Όλα τα plots και ο πίνακας αποθηκεύτηκαν στον φάκελο: {os.path.abspath(PLOTS_DIR)}")
