import pandas as pd
import matplotlib.pyplot as plt

# Leggi il primo file CSV (senza saltare righe)
df = pd.read_csv('datiPerPaper/EAD_precision_reduction.csv')

# Crea il primo grafico
plt.figure(figsize=(10, 6))

# Definisci colori e simboli per ogni linea
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

# Etichette personalizzate per le linee
labels = ["8 bit (Base model)", "6 bit", "4 bit", "2 bit"]

# Prima colonna come asse x
x = df.iloc[:, 0]

# Plotta le altre colonne come linee
for i, column in enumerate(df.columns[1:]):
    plt.plot(x, df[column], 
             marker=markers[i % len(markers)], 
             color=colors[i % len(colors)],
             label=labels[i] if i < len(labels) else column,
             linewidth=10,
             markersize=20)

plt.xlim(0, 1.0)
plt.ylim(0, 100)

plt.xlabel('Perturbation Norm', fontsize=22)
plt.ylabel('Accuracy', fontsize=22)
plt.title('Elastic-Net Adaptive Attack', fontsize=30)
plt.legend(fontsize=30)
plt.tick_params(axis='both', labelsize=18)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Leggi il secondo file CSV
df2 = pd.read_csv('datiPerPaper/Boundary_precision_reduction.csv')

# Crea il secondo grafico
plt.figure(figsize=(10, 6))

# Prima colonna come asse x
x2 = df2.iloc[:, 0]

# Plotta le altre colonne come linee
for i, column in enumerate(df2.columns[1:]):
    plt.plot(x2, df2[column], 
             marker=markers[i % len(markers)], 
             color=colors[i % len(colors)],
             label=labels[i] if i < len(labels) else column,
             linewidth=10,
             markersize=20)

plt.xlabel('Perturbation Norm', fontsize=22)
plt.ylabel('Accuracy', fontsize=22)
plt.title('Boundary Attack', fontsize=30)
plt.legend(fontsize=30)
plt.tick_params(axis='both', labelsize=18)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xlim(0, 1.0)
plt.ylim(0, 100)
plt.show()

# Leggi il terzo file CSV
df3 = pd.read_csv('datiPerPaper/Boundary_randomized.csv')

# Crea il terzo grafico
plt.figure(figsize=(10, 6))

# Etichette personalizzate per il terzo grafico
labels3 = ["Base model", "Randomized Defense (black-box)"]

# Prima colonna come asse x
x3 = df3.iloc[:, 0]

# Plotta le altre colonne come linee
for i, column in enumerate(df3.columns[1:]):
    plt.plot(x3, df3[column], 
             marker=markers[i % len(markers)], 
             color=colors[i % len(colors)],
             label=labels3[i] if i < len(labels3) else column,
             linewidth=10,
             markersize=20)

plt.xlabel('Perturbation Norm', fontsize=22)
plt.ylabel('Accuracy', fontsize=22)
plt.title('Boundary Attack', fontsize=30)
plt.legend(fontsize=30)
plt.tick_params(axis='both', labelsize=18)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xlim(0, 1.0)
plt.ylim(0, 100)
plt.show()

# Leggi il quarto file CSV
df4 = pd.read_csv('datiPerPaper/EAD_randomized.csv')

# Crea il quarto grafico
plt.figure(figsize=(10, 6))

# Etichette personalizzate per il quarto grafico
labels4 = ["Base Model", "Randomized Defense (white-box)", "Randomized Defense (gray-box)"]

# Prima colonna come asse x
x4 = df4.iloc[:, 0]

# Plotta le altre colonne come linee
for i, column in enumerate(df4.columns[1:]):
    plt.plot(x4, df4[column], 
             marker=markers[i % len(markers)], 
             color=colors[i % len(colors)],
             label=labels4[i] if i < len(labels4) else column,
             linewidth=10,
             markersize=20)

plt.xlabel('Perturbation Norm', fontsize=22)
plt.ylabel('Accuracy', fontsize=22)
plt.title('Elastic-Net Adaptive Attack', fontsize=30)
plt.legend(fontsize=30)
plt.tick_params(axis='both', labelsize=18)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xlim(0, 1.0)
plt.ylim(0, 100)
plt.show()

# Leggi il quinto file CSV
df5 = pd.read_csv('datiPerPaper/EAD_adv_training.csv')

# Crea il quinto grafico
plt.figure(figsize=(10, 6))

# Etichette personalizzate per il quinto grafico
labels5 = ["Base Model", "Adversarial Training (gray-box)", "Adversarial Training (white-box)"]

# Prima colonna come asse x
x5 = df5.iloc[:, 0]

# Plotta le altre colonne come linee
for i, column in enumerate(df5.columns[1:]):
    plt.plot(x5, df5[column], 
             marker=markers[i % len(markers)], 
             color=colors[i % len(colors)],
             label=labels5[i] if i < len(labels5) else column,
             linewidth=10,
             markersize=20)

plt.xlabel('Perturbation Norm', fontsize=22)
plt.ylabel('Accuracy', fontsize=22)
plt.title('Elastic-Net Adaptive Attack', fontsize=30)
plt.legend(fontsize=30)
plt.tick_params(axis='both', labelsize=18)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xlim(0, 1.0)
plt.ylim(0, 100)
plt.show()


# Leggi il sesto file CSV
df6 = pd.read_csv('datiPerPaper/Boundary_adv_training.csv')

# Crea il sesto grafico
plt.figure(figsize=(10, 6))

# Etichette personalizzate per il sesto grafico
labels6 = ["Base Model", "Adversarial Training (black-box)"]

# Prima colonna come asse x
x6 = df6.iloc[:, 0]

# Plotta le altre colonne come linee
for i, column in enumerate(df6.columns[1:]):
    plt.plot(x6, df6[column], 
             marker=markers[i % len(markers)], 
             color=colors[i % len(colors)],
             label=labels6[i] if i < len(labels6) else column,
             linewidth=10,
             markersize=20)

plt.xlabel('Perturbation Norm', fontsize=22)
plt.ylabel('Accuracy', fontsize=22)
plt.title('Boundary Attack', fontsize=30)
plt.legend(fontsize=30)
plt.tick_params(axis='both', labelsize=18)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xlim(0, 1.0)
plt.ylim(0, 100)
plt.show()
