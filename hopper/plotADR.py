import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

def moving_average(values, window):
    """Applica il filtro a media mobile."""
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")

def plot_multiple_runs(log_folders, output_name, plot_title):
    """
    Legge le cartelle, calcola la media e salva con nome e titolo personalizzati.
    """
    WINDOW_SIZE = 1000
    INTERPOLATE_POINTS = 1000 
    
    print(f"--- Analisi di {len(log_folders)} run ---")
    print(f"Titolo: {plot_title}")
    print(f"Output: {output_name}")

    data_x = []
    data_y = []
    min_timesteps = float('inf')

    # 1. CARICAMENTO DATI
    for folder in log_folders:
        try:
            # Carica tutti gli 8 csv della cartella e uniscili
            x, y = ts2xy(load_results(folder), "timesteps")
            
            if len(x) == 0: 
                print(f"Skipping vuota: {folder}")
                continue
            
            # Smoothing
            y_smooth = moving_average(y, window=WINDOW_SIZE)
            x_smooth = x[len(x) - len(y_smooth):]
            
            data_x.append(x_smooth)
            data_y.append(y_smooth)
            
            min_timesteps = min(min_timesteps, x_smooth[-1])
            print(f" -> OK: {folder} ({int(x_smooth[-1])} steps)")
            
        except Exception as e:
            print(f" -> ERRORE su {folder}: {e}")

    if not data_x:
        print("Nessun dato valido trovato.")
        return

    # 2. CALCOLO MEDIA (INTERPOLAZIONE)
    common_x = np.linspace(0, min_timesteps, num=INTERPOLATE_POINTS)
    interpolated_ys = []
    
    for x, y in zip(data_x, data_y):
        iy = np.interp(common_x, x, y)
        interpolated_ys.append(iy)
    
    Y_matrix = np.array(interpolated_ys)
    mean_y = np.mean(Y_matrix, axis=0)
    std_y = np.std(Y_matrix, axis=0)

    # 3. PLOTTING
    plt.figure(figsize=(10, 6))
    
    # Area ombra (Deviazione Standard)
    plt.fill_between(common_x, mean_y - std_y, mean_y + std_y, alpha=0.2, color='#0072B2', label='Std Dev')
    # Linea Media
    plt.plot(common_x, mean_y, color='#0072B2', linewidth=2, label='Mean Reward')
    
    plt.title(plot_title, fontsize=14)
    plt.xlabel("Total Timesteps", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Creazione cartella plots se non esiste
    os.makedirs("plots", exist_ok=True)
    
    # Aggiungi estensione .png se manca
    if not output_name.endswith(".png"):
        output_name += ".png"
        
    save_path = os.path.join("plots", output_name)
    plt.savefig(save_path, dpi=300)
    print(f"\nGrafico salvato con successo in: {save_path}")
    # plt.show() # Decommenta se vuoi vederlo a schermo ogni volta

if __name__ == "__main__":
    # Configurazione Argomenti da Terminale
    parser = argparse.ArgumentParser(description="Plotta la media di più run di training.")
    
    # Argomenti:
    # 1. Lista delle cartelle (obbligatorio)
    parser.add_argument("folders", nargs='+', help="Lista delle cartelle dei log (es. logs/seed1 logs/seed2)")
    
    # 2. Nome del file in uscita (opzionale, default=learning_curve)
    parser.add_argument("--name", type=str, default="learning_curve", help="Nome del file png in uscita")
    
    # 3. Titolo del grafico (opzionale, default=Average Reward)
    parser.add_argument("--title", type=str, default="Average Learning Curve", help="Titolo sopra il grafico")

    args = parser.parse_args()

    plot_multiple_runs(args.folders, args.name, args.title)