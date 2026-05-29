from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

VIDEO_SUMMARY_CSV = BASE_DIR / "outputs" / "video_summary.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

def main():
    if not VIDEO_SUMMARY_CSV.exists():
        print("\nArquivo video_summary.csv não encontrado.")
        print("Rode primeiro o script 02_gerar_videos_anotados.py")
        return

    df = pd.read_csv(VIDEO_SUMMARY_CSV)

    if df.empty:
        print("\nO arquivo video_summary.csv está vazio.")
        return

    print("\nResumo geral:")
    print(df[["video", "ground_truth", "final_prediction", "confidence_mean", "correct", "actor"]])

    accuracy = df["correct"].mean()

    print("\nAcurácia geral:")
    print(f"{accuracy:.2%}")

    # Resultado por emoção real
    by_emotion = (
        df.groupby("ground_truth")
        .agg(
            total=("video", "count"),
            acertos=("correct", "sum"),
            confianca_media=("confidence_mean", "mean")
        )
        .reset_index()
    )

    by_emotion["erros"] = by_emotion["total"] - by_emotion["acertos"]
    by_emotion["accuracy"] = by_emotion["acertos"] / by_emotion["total"]

    print("\nResultado por emoção:")
    print(by_emotion)

    # Matriz de confusão simples
    confusion_matrix = pd.crosstab(
        df["ground_truth"],
        df["final_prediction"],
        rownames=["Ground Truth"],
        colnames=["Predicted"],
        dropna=False
    )

    print("\nMatriz de confusão:")
    print(confusion_matrix)

    # Lista de erros
    errors = df[df["correct"] == False][
        ["video", "ground_truth", "final_prediction", "confidence_mean", "actor"]
    ]

    print("\nErros encontrados:")
    print(errors)

    # Salvar arquivos
    by_emotion_path = OUTPUT_DIR / "results_by_emotion.csv"
    confusion_matrix_path = OUTPUT_DIR / "confusion_matrix.csv"
    errors_path = OUTPUT_DIR / "errors.csv"

    by_emotion.to_csv(by_emotion_path, index=False)
    confusion_matrix.to_csv(confusion_matrix_path)
    errors.to_csv(errors_path, index=False)

    print("\nArquivos gerados:")
    print(f"- {by_emotion_path}")
    print(f"- {confusion_matrix_path}")
    print(f"- {errors_path}")

    print("\nResumo para apresentação:")
    print(
        f"O pipeline processou {len(df)} vídeos do RAVDESS, "
        f"com acurácia geral de {accuracy:.2%}. "
        f"Foram gerados resultados por emoção, matriz de confusão "
        f"e uma lista de erros para análise qualitativa."
    )


if __name__ == "__main__":
    main()