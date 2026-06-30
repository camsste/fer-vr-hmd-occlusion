import cv2
import mediapipe as mp
import numpy as np
import os
import math
from tqdm import tqdm

# Inicialização do MediaPipe
mp_face_mesh = mp.solutions.face_mesh

def aplicar_hmd_video(caminho_entrada, diretorio_saida):
    os.makedirs(diretorio_saida, exist_ok=True)
    nome_arquivo = os.path.basename(caminho_entrada)
    caminho_saida = os.path.join(diretorio_saida, f"HMD_{nome_arquivo}")

    cap = cv2.VideoCapture(caminho_entrada)
    if not cap.isOpened():
        print(f"[ERRO] Não abriu o vídeo: {caminho_entrada}")
        return

    # Extrai as propriedades do vídeo original
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(caminho_saida, fourcc, fps, (w, h))

    print(f"\n[INFO] Renderizando HMD (Bloco Rígido) em: {nome_arquivo}")
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        pbar = tqdm(total=total_frames, desc="Renderizando Frames", unit="frame")
        
        while cap.isOpened():
            sucesso, frame = cap.read()
            if not sucesso:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = face_mesh.process(frame_rgb)
            
            if resultados.multi_face_landmarks:
                landmarks = resultados.multi_face_landmarks[0].landmark
                
                # Pegamos apenas 4 âncoras extremas
                pt_esq = landmarks[234]  # Extremo esquerdo da bochecha
                pt_dir = landmarks[454]  # Extremo direito da bochecha
                pt_topo = landmarks[10]  # Topo da testa
                pt_base = landmarks[4]   # Ponta do nariz
                
                # Conversão para pixels na tela
                x_esq, y_esq = int(pt_esq.x * w), int(pt_esq.y * h)
                x_dir, y_dir = int(pt_dir.x * w), int(pt_dir.y * h)
                x_topo, y_topo = int(pt_topo.x * w), int(pt_topo.y * h)
                x_base, y_base = int(pt_base.x * w), int(pt_base.y * h)

                # Encontra o centro exato da "caixa" do HMD
                centro_x = int((x_esq + x_dir) / 2)
                centro_y = int((y_topo + y_base) / 2)

                # Calcula Largura e Altura baseadas na anatomia, com expansão (Padding)
                # Multiplicar a largura por 1.3 faz o HMD "vazar" 30% pelas laterais da cabeça
                largura = int(math.hypot(x_dir - x_esq, y_dir - y_esq) * 1.30)
                altura = int(math.hypot(x_base - x_topo, y_base - y_topo) * 1.15)

                # Calcula o ângulo de inclinação da cabeça 
                angulo_rad = math.atan2(y_dir - y_esq, x_dir - x_esq)
                angulo_graus = math.degrees(angulo_rad)

                # Monta a geometria do retângulo rotacionado no formato do OpenCV
                retangulo = ((centro_x, centro_y), (largura, altura), angulo_graus)
                box = cv2.boxPoints(retangulo)
                box = np.intp(box) 
                
                # Desenha o bloco maciço sobre o rosto
                COR_HMD = (15, 15, 15) # Preto escuro, levemente fosco
                cv2.fillPoly(frame, [box], COR_HMD)
                
            out.write(frame)
            pbar.update(1)
            
    pbar.close()
    cap.release()
    out.release()
    print(f"\n[SUCESSO] Vídeo salvo em: {caminho_saida}\n")

if __name__ == "__main__":
    video_teste = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\dataset\RAVDESS\Video_Song_Actor_02\Actor_02\01-02-01-01-01-01-02.mp4"
    pasta_saida = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\dataset\saida_hmd"
    
    aplicar_hmd_video(video_teste, pasta_saida)