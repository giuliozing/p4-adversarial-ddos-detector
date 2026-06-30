import json
import matplotlib.pyplot as plt
import os

def plot_attack_resistance(file_path):
    # Apri il file JSON e carica i dati
    with open(file_path, 'r') as json_file:
        resultVector = json.load(json_file)
    
    # Estrai le accuracy per gli attacchi FGSM, PGD e CW
    accuracies = {
        "FGSM": [result[1] for result in resultVector["FGSM"]] if "FGSM" in resultVector else None,
        "Noise": [result[1] for result in resultVector["Noise"]] if "Noise" in resultVector else None,
    }
    
    # Estrai le accuracy per DeepFool e JSMA
    deepfool_accuracy = resultVector["DeepFool"][0][1] if "DeepFool" in resultVector else None
    jsma_accuracy = resultVector["JSMA"][0][1] if "JSMA" in resultVector else None
    pgd_accuracy = resultVector["PGD"][0][1] if "PGD" in resultVector else None
    # Grafica le accuracy
    plt.figure(figsize=(10, 6))
    
    for attack, acc in accuracies.items():
        if acc is not None:
            plt.plot(acc, label=attack)
    
    # Aggiungi linee orizzontali per Baseline, DeepFool e JSMA
    
    if accuracies["FGSM"] is not None: 
        plt.axhline(y=accuracies["FGSM"][0], color='g', linestyle='--', label='Baseline')
    if pgd_accuracy is not None: 
        plt.axhline(y=pgd_accuracy, color='r', linestyle='--', label='PGD')
    if deepfool_accuracy is not None:
        plt.axhline(y=deepfool_accuracy, color='r', linestyle='--', label='DeepFool')
    if jsma_accuracy is not None:
        plt.axhline(y=jsma_accuracy, color='b', linestyle='--', label='JSMA')
    
    if accuracies["FGSM"] is not None:
        plt.xlabel("Epsilon, values from 0 to " + str((len(accuracies["FGSM"])-1)/10))
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    plt.title(file_path)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Prendi in input da tastiera il nome del file JSON
    file_path = input("Inserisci il nome del file JSON: ")
    file_path = os.path.join('performance', file_path)
    plot_attack_resistance(file_path)