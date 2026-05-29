from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]

VIDEO_SUMMARY_CSV = BASE_DIR / "outputs" / "video_summary.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not VIDEO_SUMMARY_CSV.exists():
        print("\nArquivo video_summary.csv não encontrado.")
        print("Rode primeiro o script 02_gerar_videos_anotados.py")
        return

    df = pd.read_csv(VIDEO_SUMMARY_CSV)

    if df.empty:
        print("\nO arquivo video_summary.csv está vazio.")
        return

    labels = sorted(set(df["ground_truth"]).union(set(df["final_prediction"])))

    confusion_matrix = pd.crosstab(
        df["ground_truth"],
        df["final_prediction"],
        rownames=["Ground Truth"],
        colnames=["Predicted"],
        dropna=False
    )

    confusion_matrix = confusion_matrix.reindex(
        index=labels,
        columns=labels,
        fill_value=0
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(confusion_matrix.values)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted Emotion")
    ax.set_ylabel("Ground Truth Emotion")
    ax.set_title("Matriz de Confusão - RAVDESS Frame-by-Frame Baseline")

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = confusion_matrix.values[i, j]
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center"
            )

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    output_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("\nMatriz de confusão gerada com sucesso.")
    print(f"Imagem salva em: {output_path}")

    print("\nMatriz usada:")
    print(confusion_matrix)


if __name__ == "__main__":
    main()