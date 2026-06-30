import cv2
import mediapipe as mp
import numpy as np
import os
import math
import shutil
from tqdm import tqdm

# Inicialização do MediaPipe (Fora do loop para performance)
mp_face_mesh = mp.solutions.face_mesh

def processar_video(caminho_entrada, caminho_saida, com_olhos=False):
    cap = cv2.VideoCapture(caminho_entrada)
    if not cap.isOpened(): return

    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(caminho_saida, fourcc, fps, (w, h))

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            sucesso, frame = cap.read()
            if not sucesso: break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = face_mesh.process(frame_rgb)
            
            if resultados.multi_face_landmarks:
                landmarks = resultados.multi_face_landmarks[0].landmark
                
                # Ancoragem Rígida (Crânio)
                pt_esq, pt_dir = landmarks[234], landmarks[454]
                pt_centro = landmarks[168]
                x_esq, y_esq = int(pt_esq.x * w), int(pt_esq.y * h)
                x_dir, y_dir = int(pt_dir.x * w), int(pt_dir.y * h)
                x_c, y_c = int(pt_centro.x * w), int(pt_centro.y * h)

                dist_cranio = math.hypot(x_dir - x_esq, y_dir - y_esq)
                largura, altura = int(dist_cranio * 1.30), int(dist_cranio * 0.70)
                angulo_rad = math.atan2(y_dir - y_esq, x_dir - x_esq)
                angulo_graus = math.degrees(angulo_rad)

                # Desenha HMD Preto
                retangulo = ((x_c, y_c), (largura, altura), angulo_graus)
                box = np.intp(cv2.boxPoints(retangulo))
                cv2.fillPoly(frame, [box], (0, 0, 0))

                # Desenha Olhos
                if com_olhos:
                    px_esq = (int(landmarks[468].x * w), int(landmarks[468].y * h))
                    px_dir = (int(landmarks[473].x * w), int(landmarks[473].y * h))
                    eixo_x, eixo_y = int(largura * 0.08), int(largura * 0.035)
                    raio_iris = int(eixo_y * 0.8)
                    cv2.ellipse(frame, px_esq, (eixo_x, eixo_y), angulo_graus, 0, 360, (255, 255, 255), -1)
                    cv2.circle(frame, px_esq, raio_iris, (0, 0, 0), -1)
                    cv2.ellipse(frame, px_dir, (eixo_x, eixo_y), angulo_graus, 0, 360, (255, 255, 255), -1)
                    cv2.circle(frame, px_dir, raio_iris, (0, 0, 0), -1)
                
            out.write(frame)
    cap.release()
    out.release()

def rodar_pipeline():
    PASTA_ORIGEM = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\dataset\RAVDESS"
    PASTA_DESTINO = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\dataset_processado"

    print("Iniciando varredura do dataset...")
    
    for root, dirs, files in os.walk(PASTA_ORIGEM):
        for file in files:
            if file.endswith(".mp4"):
                caminho_full = os.path.join(root, file)
                rel_path = os.path.relpath(root, PASTA_ORIGEM)
                pasta_saida_final = os.path.join(PASTA_DESTINO, rel_path)
                os.makedirs(pasta_saida_final, exist_ok=True)

                # Separando nome da extensão para manipular o sufixo
                nome_base, extensao = os.path.splitext(file)

                print(f"Processando: {file}")
                
                # 1. Copia o Original
                shutil.copy2(caminho_full, os.path.join(pasta_saida_final, file))
                
                # 2. Gera HMD (Sufixo no final)
                processar_video(caminho_full, os.path.join(pasta_saida_final, f"{nome_base}_HMD{extensao}"), com_olhos=False)
                
                # 3. Gera HMD + Olhos (Sufixo no final)
                processar_video(caminho_full, os.path.join(pasta_saida_final, f"{nome_base}_HMD_Olhos{extensao}"), com_olhos=True)

if __name__ == "__main__":
    rodar_pipeline()
    print("\nDataset processado completamente! Tudo pronto para a classificação.")