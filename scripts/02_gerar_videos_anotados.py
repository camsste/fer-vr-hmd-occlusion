from pathlib import Path
from collections import Counter

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from deepface import DeepFace


# =========================
# CONFIGURAÇÕES
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

SAMPLES_CSV = BASE_DIR / "outputs" / "selected_10_samples.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
ANNOTATED_DIR = OUTPUT_DIR / "annotated_videos"

ANALYZE_EVERY_N_FRAMES = 5


# =========================
# FUNÇÕES AUXILIARES
# =========================

def normalize_prediction(prediction: str) -> str:
    """
    Ajusta os nomes das emoções do DeepFace para comparar com o RAVDESS.
    """

    if prediction == "fear":
        return "fearful"

    if prediction == "surprise":
        return "surprised"

    return prediction


def resolve_video_path(row):
    """
    Tenta encontrar o caminho real do vídeo.

    O CSV pode guardar caminhos relativos como:
    ../dataset/RAVDESS/...
    ..\\dataset\\RAVDESS\\...

    Então testamos várias possibilidades.
    """

    raw_path = str(row["path"])
    video_name = str(row["video"])

    possible_paths = [
        Path(raw_path),
        SCRIPT_DIR / raw_path,
        BASE_DIR / raw_path,
        BASE_DIR / "dataset" / "RAVDESS" / video_name,
    ]

    # Caso o arquivo esteja dentro das pastas Actor_XX
    dataset_dir = BASE_DIR / "dataset" / "RAVDESS"

    if dataset_dir.exists():
        matches = list(dataset_dir.rglob(video_name))

        for match in matches:
            possible_paths.append(match)

    for path in possible_paths:
        resolved = path.resolve()

        if resolved.exists():
            return resolved

    print("\nNão encontrei o arquivo do vídeo.")
    print(f"Vídeo: {video_name}")
    print("Caminhos testados:")

    for path in possible_paths:
        print(f"- {path.resolve()}")

    return None


def draw_annotation(frame, true_emotion, predicted_emotion, confidence):
    """
    Desenha no vídeo:
    - Ground Truth
    - Predição
    - Confiança
    """

    cv2.rectangle(frame, (10, 10), (760, 125), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Ground Truth: {true_emotion}",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Predicted: {predicted_emotion}",
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.1f}%",
        (25, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    return frame


def analyze_frame(frame):
    """
    Analisa um frame com DeepFace.
    """

    analysis = DeepFace.analyze(
        img_path=frame,
        actions=["emotion"],
        enforce_detection=False,
        detector_backend="opencv"
    )

    if isinstance(analysis, list):
        analysis = analysis[0]

    raw_predicted_emotion = analysis.get("dominant_emotion", "unknown")
    predicted_emotion = normalize_prediction(raw_predicted_emotion)

    emotion_probs = analysis.get("emotion", {})
    confidence = float(emotion_probs.get(raw_predicted_emotion, 0.0))

    normalized_probs = {}

    for emotion_name, prob in emotion_probs.items():
        normalized_name = normalize_prediction(emotion_name)
        normalized_probs[normalized_name] = prob

    return predicted_emotion, confidence, normalized_probs


def process_video(row):
    """
    Processa um vídeo:
    - abre o arquivo
    - roda emoção frame a frame
    - gera vídeo anotado
    - retorna resultados por frame e resumo por vídeo
    """

    video_path = resolve_video_path(row)

    if video_path is None:
        return [], None

    true_emotion = row["ground_truth"]

    output_video_path = ANNOTATED_DIR / f"{video_path.stem}_annotated.mp4"

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"\nOpenCV não conseguiu abrir o vídeo: {video_path}")
        return [], None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames == 0 or width == 0 or height == 0:
        print(f"\nVídeo inválido ou corrompido: {video_path}")
        cap.release()
        return [], None

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_rows = []

    last_predicted_emotion = "unknown"
    last_confidence = 0.0
    last_probs = {}

    predictions_for_video = []
    confidence_values = []

    for frame_idx in tqdm(
        range(total_frames),
        desc=f"Processando {video_path.name}"
    ):
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % ANALYZE_EVERY_N_FRAMES == 0:
            try:
                predicted_emotion, confidence, probs = analyze_frame(frame)

                last_predicted_emotion = predicted_emotion
                last_confidence = confidence
                last_probs = probs

                predictions_for_video.append(predicted_emotion)
                confidence_values.append(confidence)

            except Exception as e:
                last_predicted_emotion = "error"
                last_confidence = 0.0
                last_probs = {}

        annotated_frame = draw_annotation(
            frame=frame,
            true_emotion=true_emotion,
            predicted_emotion=last_predicted_emotion,
            confidence=last_confidence
        )

        writer.write(annotated_frame)

        frame_row = {
            "video": row["video"],
            "frame": frame_idx,
            "ground_truth": true_emotion,
            "predicted_emotion": last_predicted_emotion,
            "confidence": last_confidence,
            "modality": row["modality"],
            "channel": row["channel"],
            "emotion_code": row["emotion_code"],
            "intensity_code": row["intensity_code"],
            "statement_code": row["statement_code"],
            "repetition_code": row["repetition_code"],
            "actor": row["actor"]
        }

        for emotion_name, prob in last_probs.items():
            frame_row[f"prob_{emotion_name}"] = prob

        frame_rows.append(frame_row)

    cap.release()
    writer.release()

    if len(predictions_for_video) > 0:
        final_prediction = Counter(predictions_for_video).most_common(1)[0][0]
        confidence_mean = float(np.mean(confidence_values))
    else:
        final_prediction = "unknown"
        confidence_mean = 0.0

    summary_row = {
        "video": row["video"],
        "video_path": str(video_path),
        "output_video": str(output_video_path),
        "ground_truth": true_emotion,
        "final_prediction": final_prediction,
        "confidence_mean": confidence_mean,
        "correct": final_prediction == true_emotion,
        "modality": row["modality"],
        "channel": row["channel"],
        "emotion_code": row["emotion_code"],
        "intensity_code": row["intensity_code"],
        "statement_code": row["statement_code"],
        "repetition_code": row["repetition_code"],
        "actor": row["actor"]
    }

    return frame_rows, summary_row


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    if not SAMPLES_CSV.exists():
        print("\nArquivo selected_10_samples.csv não encontrado.")
        print(f"Esperado em: {SAMPLES_CSV}")
        print("Rode primeiro o script 01_listar_amostras.py")
        return

    samples = pd.read_csv(SAMPLES_CSV)

    print("\nAmostras carregadas:")
    print(samples[["video", "ground_truth", "actor", "modality", "channel"]])

    all_frame_rows = []
    summary_rows = []

    for _, row in samples.iterrows():
        frame_rows, summary_row = process_video(row)

        all_frame_rows.extend(frame_rows)

        if summary_row is not None:
            summary_rows.append(summary_row)

    frame_results = pd.DataFrame(all_frame_rows)
    video_summary = pd.DataFrame(summary_rows)

    frame_results_path = OUTPUT_DIR / "frame_results.csv"
    video_summary_path = OUTPUT_DIR / "video_summary.csv"

    frame_results.to_csv(frame_results_path, index=False)
    video_summary.to_csv(video_summary_path, index=False)

    print("\nProcessamento finalizado.")
    print(f"Resultados por frame salvos em: {frame_results_path}")
    print(f"Resumo por vídeo salvo em: {video_summary_path}")
    print(f"Vídeos anotados salvos em: {ANNOTATED_DIR}")

    print("\nResumo dos vídeos:")

    if len(video_summary) > 0:
        print(
            video_summary[
                [
                    "video",
                    "ground_truth",
                    "final_prediction",
                    "confidence_mean",
                    "correct",
                    "actor"
                ]
            ]
        )

        accuracy = video_summary["correct"].mean()
        print(f"\nAcurácia nas 10 amostras: {accuracy:.2%}")

    else:
        print("Nenhum vídeo foi processado.")
        print("Verifique se a pasta dataset/RAVDESS está no lugar certo.")


if __name__ == "__main__":
    main()