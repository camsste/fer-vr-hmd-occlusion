from pathlib import Path
import pandas as pd

# Caminho do dataset a partir da pasta scripts/
DATASET_DIR = Path("../dataset/RAVDESS")

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

MODALITY_MAP = {
    "01": "audio-video",
    "02": "video-only",
    "03": "audio-only"
}

CHANNEL_MAP = {
    "01": "speech",
    "02": "song"
}


def parse_ravdess_filename(video_path):
    """
    Lê o nome do arquivo do RAVDESS e extrai os metadados.

    Formato:
    Modality-Channel-Emotion-Intensity-Statement-Repetition-Actor

    Exemplo:
    01-01-03-02-01-01-01.mp4

    01 = audio-video
    01 = speech
    03 = happy
    02 = strong intensity
    01 = statement
    01 = repetition
    01 = actor
    """

    parts = video_path.stem.split("-")

    if len(parts) != 7:
        return None

    modality, channel, emotion, intensity, statement, repetition, actor = parts

    return {
        "video": video_path.name,
        "path": str(video_path),
        "modality_code": modality,
        "modality": MODALITY_MAP.get(modality, "unknown"),
        "channel_code": channel,
        "channel": CHANNEL_MAP.get(channel, "unknown"),
        "emotion_code": emotion,
        "ground_truth": EMOTION_MAP.get(emotion, "unknown"),
        "intensity_code": intensity,
        "statement_code": statement,
        "repetition_code": repetition,
        "actor": int(actor)
    }


def main():
    videos = []

    # Procura todos os arquivos .mp4 dentro da pasta do dataset
    for video_path in DATASET_DIR.rglob("*.mp4"):
        metadata = parse_ravdess_filename(video_path)

        if metadata is None:
            continue

        # Por enquanto vamos usar apenas:
        # modality_code 01 = audio-video
        # channel_code 01 = speech
        if metadata["modality_code"] == "01" and metadata["channel_code"] == "01":
            videos.append(metadata)

    df = pd.DataFrame(videos)

    if df.empty:
        print("\nNenhum vídeo foi encontrado.")
        print("Verifique se o caminho DATASET_DIR está correto:")
        print(DATASET_DIR)
        return

    print("\nTotal de vídeos audio-video/speech encontrados:")
    print(len(df))

    print("\nDistribuição por emoção:")
    print(df["ground_truth"].value_counts())

    # Seleciona 10 amostras tentando variar emoção e ator
    samples_list = []
    used_actors = set()

    # Primeiro: tenta pegar uma amostra por emoção, com atores diferentes
    for emotion in df["ground_truth"].unique():
        candidates = df[
            (df["ground_truth"] == emotion) &
            (~df["actor"].isin(used_actors))
        ]

        # Se não encontrar ator novo para aquela emoção, pega qualquer um da emoção
        if len(candidates) == 0:
            candidates = df[df["ground_truth"] == emotion]

        sample = candidates.sample(1, random_state=42)
        samples_list.append(sample)

        used_actors.add(sample.iloc[0]["actor"])

    samples = pd.concat(samples_list, ignore_index=True)

    # Se ainda tiver menos de 10, completa com outros atores ainda não usados
    if len(samples) < 10:
        remaining = df[
            ~df["video"].isin(samples["video"]) &
            ~df["actor"].isin(used_actors)
        ]

        needed = 10 - len(samples)

        if len(remaining) > 0:
            extra = remaining.sample(
                min(needed, len(remaining)),
                random_state=123
            )

            samples = pd.concat([samples, extra], ignore_index=True)

            for actor in extra["actor"]:
                used_actors.add(actor)

    # Se ainda tiver menos de 10, completa com qualquer vídeo restante
    if len(samples) < 10:
        remaining = df[~df["video"].isin(samples["video"])]

        needed = 10 - len(samples)

        if len(remaining) > 0:
            extra = remaining.sample(
                min(needed, len(remaining)),
                random_state=456
            )

            samples = pd.concat([samples, extra], ignore_index=True)

    # Limita a 10 amostras
    samples = samples.head(10)

    print("\n10 amostras selecionadas com variação de atores:")
    print(samples[["video", "ground_truth", "modality", "channel", "actor"]])

    print("\nQuantidade de atores diferentes:")
    print(samples["actor"].nunique())

    print("\nAtores selecionados:")
    print(samples["actor"].tolist())

    print("\nEmoções selecionadas:")
    print(samples["ground_truth"].tolist())

    # Salva o CSV
    output_dir = Path("../outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "selected_10_samples.csv"
    samples.to_csv(output_path, index=False)

    print(f"\nArquivo criado em: {output_path}")


if __name__ == "__main__":
    main()