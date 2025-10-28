import os #εργασία με αρχεία και φακέλους
import re #regex για εξαγωγή μετρικών
import pandas as pd #διαχείριση δεδομένων
import matplotlib.pyplot as plt #δημιουργία διαγραμμάτων
import seaborn as sns #βελτιωμένη απεικόνιση
from matplotlib.backends.backend_pdf import PdfPages #εξαγωγή σε PDF

#ρυθμίσεις
RESULTS_DIR = "."         #ίδιος φάκελος με το script
PLOTS_DIR = "plots"       #αποθήκευση διαγραμμάτων
os.makedirs(PLOTS_DIR, exist_ok=True) #δημιουργία φακέλου αν δεν υπάρχει

#ρegex patterns για τις μετρικές
metrics_pattern = {
    "Average AF": r"Average AF:\s*([0-9.eE+-]+)",
    "Recall@N": r"Recall@N:\s*([0-9.eE+-]+)",
    "QPS": r"QPS:\s*([0-9.eE+-]+)",
    "tApproximateAverage": r"tApproximateAverage:\s*([0-9.eE+-]+)",
    "tTrueAverage": r"tTrueAverage:\s*([0-9.eE+-]+)"
}

#συνάρτηση ανάγνωσης ενός αρχείου αποτελεσμάτων
def parse_results_file(filepath):
    with open(filepath, "r") as f:
        text = f.read() #ολόκληρο το περιεχόμενο

    data = {} #αποθήκευση μετρικών
    for key, pattern in metrics_pattern.items():
        match = re.search(pattern, text) #αναζήτηση με regex
        data[key] = float(match.group(1)) if match else None #αποθήκευση τιμής ή None

    filename = os.path.basename(filepath).lower() #όνομα αρχείου για εντοπισμό μεθόδου/dataset

    #εντοπισμός μεθόδου
    for method in ["lsh", "hypercube", "ivfflat", "ivfpq"]:
        if method in filename:
            data["Method"] = method.upper()  #μετατροπή σε κεφαλαία
            break
    else:
        data["Method"] = "Unknown"

    #εντοπισμός dataset
    for dataset in ["mnist", "sift"]:
        if dataset in filename:
            data["Dataset"] = dataset.upper() #μετατροπή σε κεφαλαία
            break
    else:
        data["Dataset"] = "Unknown"

    return data


#ανάγνωση όλων των αποτελεσμάτων
records = []
for fname in os.listdir(RESULTS_DIR): #περιήγηση στον φάκελο
    if fname.startswith("results_") and fname.endswith(".txt"):
        print(f"Reading: {fname}") #ενημέρωση χρήστη
        filepath = os.path.join(RESULTS_DIR, fname) #πλήρης διαδρομή
        records.append(parse_results_file(filepath)) #ανάγνωση και αποθήκευση

if not records:
    print("Δεν βρέθηκαν αρχεία results_*.txt στον φάκελο.")
    exit(1)

df = pd.DataFrame(records)
print("\n Σύνοψη αποτελεσμάτων:\n")
print(df) #εκτύπωση πίνακα αποτελεσμάτων

#ομαδοποίηση (αν υπάρχουν πολλαπλά runs για ίδια μέθοδο/dataset)
df_grouped = df.groupby(["Dataset", "Method"], as_index=False).mean()

#plot configuration
sns.set(style="whitegrid", font_scale=1.15)

def plot_metric(metric, title, ylabel):
    plt.figure(figsize=(7, 5)) #μέγεθος διαγράμματος
    ax = sns.barplot(data=df_grouped, x="Dataset", y=metric, hue="Method", palette="Set2") #μπαρ πλοτ
    plt.title(title) #τίτλος
    plt.ylabel(ylabel) #ετικέτα άξονα y
    plt.xlabel("Dataset") #ετικέτα άξονα x
    plt.legend(title="Method", loc="best") #υπόμνημα

    #εμφάνιση τιμών πάνω στις μπάρες
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9) #μορφοποίηση τιμών

    plt.tight_layout() #βελτιστοποίηση layout
    safe_name = metric.replace("@", "_at_").replace(" ", "_") #ασφαλές όνομα αρχείου
    plt.savefig(f"{PLOTS_DIR}/{safe_name}.png", dpi=150) #αποθήκευση εικόνας
    return plt.gcf()  #επιστρέφει το figure object για PDF export


#δημιουργία όλων των διαγραμμάτων + PDF export
plots = []
plots.append(plot_metric("Average AF", "Average Approximation Factor", "AF"))
plots.append(plot_metric("Recall@N", "Recall@N Comparison", "Recall"))
plots.append(plot_metric("QPS", "Queries Per Second", "QPS"))
plots.append(plot_metric("tApproximateAverage", "Average Approximate Query Time (ms)", "ms"))
plots.append(plot_metric("tTrueAverage", "Average True NN Time (ms)", "ms"))

#extra scatter: Trade-off ταχύτητας/ποιότητας
plt.figure(figsize=(7,5))
sns.scatterplot( 
    data=df_grouped, x="QPS", y="Average AF",
    hue="Method", style="Dataset", s=120, palette="Set1"
)
plt.title("Trade-off: Speed vs Approximation Quality") #τίτλος
plt.xlabel("Queries Per Second (QPS)") #ετικέτα άξονα x
plt.tight_layout() #βελτιστοποίηση layout
plt.savefig(f"{PLOTS_DIR}/tradeoff_speed_vs_quality.png", dpi=150) #αποθήκευση εικόνας
plots.append(plt.gcf()) #προσθήκη στο PDF

#εξαγωγή όλων σε ένα PDF
pdf_path = os.path.join(PLOTS_DIR, "summary_report.pdf") #αποθήκευση PDF
with PdfPages(pdf_path) as pdf:
    for fig in plots:
        pdf.savefig(fig) #προσθήκη κάθε διαγράμματος στο PDF
print(f"\n Summary report saved: {pdf_path}")

print(f"\n Όλα τα plots αποθηκεύτηκαν στον φάκελο: {os.path.abspath(PLOTS_DIR)}")
