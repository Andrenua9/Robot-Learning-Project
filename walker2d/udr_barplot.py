import matplotlib.pyplot as plt
import numpy as np

def create_thesis_plot():
    # --- 1. I DATI (Inseriti manualmente dai risultati [results.md]) ---
    labels = ['UDR Asym 0.30\n(Best)', 'UDR Sym 0.30', 'UDR Sym 0.50\n(Unstable)']
    
    # Medie
    source_means = [2566.73, 2517.61, 1676.20]
    target_means = [2241.73, 2185.05, 967.86]
    
    # Deviazioni Standard (Error Bars)
    source_std = [110.14, 65.26, 100.11]
    target_std = [164.63, 254.02, 509.28]

    # --- 2. SETUP DEL GRAFICO ---
    x = np.arange(len(labels))  # Posizione delle etichette
    width = 0.35  # Larghezza delle barre

    fig, ax = plt.subplots(figsize=(10, 6)) # Dimensione immagine (10x6 pollici)

    # Creazione delle barre
    # 'capsize' crea il trattino orizzontale sopra la barra di errore
    rects1 = ax.bar(x - width/2, source_means, width, 
                    yerr=source_std, label='Source Env (Training)', 
                    capsize=5, color='#4c72b0', alpha=0.9, edgecolor='black')
    
    rects2 = ax.bar(x + width/2, target_means, width, 
                    yerr=target_std, label='Target Env (Transfer)', 
                    capsize=5, color='#dd8452', alpha=0.9, edgecolor='black')

    # --- 3. FORMATTAZIONE ESTETICA ---
    ax.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
    ax.set_title('Robustness Comparison: Symmetric vs Asymmetric UDR', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    
    # Griglia orizzontale leggera per facilitare la lettura
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True) # Mette la griglia dietro le barre

    # Limite asse Y (opzionale, per dare aria sopra)
    ax.set_ylim(0, 3000)

    # --- 4. AGGIUNTA ETICHETTE VALORI SOPRA LE BARRE ---
    def autolabel(rects):
        """Attacca un'etichetta di testo sopra ogni barra, visualizzando l'altezza."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.0f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    # --- 5. SALVATAGGIO ---
    plt.tight_layout()
    
    # Salva in PDF (Vettoriale, perfetto per LaTeX/Word) e PNG (per preview)
    plt.savefig('udr_results_comparison.pdf', format='pdf', dpi=300)
    plt.savefig('udr_results_comparison.png', format='png', dpi=300)
    
    print("Grafici salvati come 'udr_results_comparison.pdf' e '.png'")
    plt.show()

if __name__ == "__main__":
    create_thesis_plot()