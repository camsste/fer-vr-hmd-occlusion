import cv2
import mediapipe as mp
import numpy as np
import os
import math
from tqdm import tqdm

# Inicialização do MediaPipe
mp_face_mesh = mp.solutions.face_mesh

def aplicar_hmd_com_olhos(caminho_entrada, diretorio_saida):
    os.makedirs(diretorio_saida, exist_ok=True)
    nome_arquivo = os.path.basename(caminho_entrada)
    caminho_saida = os.path.join(diretorio_saida, f"HMD_Olhos_{nome_arquivo}")

    cap = cv2.VideoCapture(caminho_entrada)
    if not cap.isOpened():
        print(f"[ERRO] Não abriu o vídeo: {caminho_entrada}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(caminho_saida, fourcc, fps, (w, h))

    print(f"\n[INFO] Injetando HMD Estático + Olhos Realistas em: {nome_arquivo}")
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        pbar = tqdm(total=total_frames, desc="Renderizando", unit="frame")
        
        while cap.isOpened():
            sucesso, frame = cap.read()
            if not sucesso:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = face_mesh.process(frame_rgb)
            
            if resultados.multi_face_landmarks:
                landmarks = resultados.multi_face_landmarks[0].landmark
                
                # --- 1. MATEMÁTICA DO HMD (Ancoragem de Crânio Rígida) ---
                # Usamos os extremos do rosto (têmporas) e a ponte do nariz, 
                # pois são os únicos ossos que não se movem quando a boca abre.
                pt_esq = landmarks[234]
                pt_dir = landmarks[454]
                pt_centro_nariz = landmarks[168] 
                
                x_esq, y_esq = int(pt_esq.x * w), int(pt_esq.y * h)
                x_dir, y_dir = int(pt_dir.x * w), int(pt_dir.y * h)
                x_centro, y_centro = int(pt_centro_nariz.x * w), int(pt_centro_nariz.y * h)

                # Distância rígida do crânio (Não deforma com a fala)
                dist_cranio = math.hypot(x_dir - x_esq, y_dir - y_esq)

                # Travamos a largura e a altura baseadas nessa distância óssea
                largura = int(dist_cranio * 1.30)
                altura = int(dist_cranio * 0.70) # Altura proporcional travada
                
                # Ângulo de rotação da cabeça
                angulo_rad = math.atan2(y_dir - y_esq, x_dir - x_esq)
                angulo_graus = math.degrees(angulo_rad)

                # Montagem geométrica centrada na ponte do nariz
                retangulo = ((x_centro, y_centro), (largura, altura), angulo_graus)
                box = cv2.boxPoints(retangulo)
                box = np.intp(box) 
                
                # Desenha o bloco com Preto Absoluto (Vantablack)
                COR_HMD = (0, 0, 0)
                cv2.fillPoly(frame, [box], COR_HMD)

                # --- 2. MATEMÁTICA DOS OLHOS REALISTAS (Estáticos) ---
                p_esq = landmarks[468] 
                p_dir = landmarks[473]

                px_esq = (int(p_esq.x * w), int(p_esq.y * h))
                px_dir = (int(p_dir.x * w), int(p_dir.y * h))

                # O tamanho dos olhos agora é calculado pela largura travada, não sofrem "efeito sanfona"
                eixo_x_esclera = int(largura * 0.08)  
                eixo_y_esclera = int(largura * 0.035) 
                raio_iris = int(eixo_y_esclera * 0.8) 
                
                # Desenho Olho Esquerdo
                cv2.ellipse(frame, px_esq, (eixo_x_esclera, eixo_y_esclera), angulo_graus, 0, 360, (255, 255, 255), -1)
                cv2.circle(frame, px_esq, raio_iris, (0, 0, 0), -1) # Pupila preta absoluta

                # Desenho Olho Direito
                cv2.ellipse(frame, px_dir, (eixo_x_esclera, eixo_y_esclera), angulo_graus, 0, 360, (255, 255, 255), -1)
                cv2.circle(frame, px_dir, raio_iris, (0, 0, 0), -1) # Pupila preta absoluta
                
            out.write(frame)
            pbar.update(1)
            
    pbar.close()
    cap.release()
    out.release()
    print(f"\n[SUCESSO] Vídeo com ancoragem rígida salvo em: {caminho_saida}\n")

if __name__ == "__main__":
    video_teste = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\dataset\RAVDESS\Video_Song_Actor_02\Actor_02\01-02-01-01-01-01-02.mp4"
    pasta_saida = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\dataset\saida_hmd"
    
    aplicar_hmd_com_olhos(video_teste, pasta_saida)