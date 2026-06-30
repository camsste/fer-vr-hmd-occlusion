import cv2
import mediapipe as mp
import numpy as np
import os
import math
import pandas as pd
from tqdm import tqdm

# Dicionário de emoções do RAF-DB (Padrão Oficial)
EMOCOES_RAF = {
    1: '1_Surprise',
    2: '2_Fear',
    3: '3_Disgust',
    4: '4_Happiness',
    5: '5_Sadness',
    6: '6_Anger',
    7: '7_Neutral'
}

# Inicialização do MediaPipe (Modo Imagem Estática)
mp_face_mesh = mp.solutions.face_mesh

def processar_imagem_rafdb(caminho_img_origem, pasta_baseline, pasta_hmd, pasta_hmd_olhos, nome_arquivo, face_mesh):
    img = cv2.imread(caminho_img_origem)
    if img is None:
        return False

    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    resultados = face_mesh.process(img_rgb)
    
    if resultados.multi_face_landmarks:
        # --- SUCESSO: Salva tudo com o nome original ---
        cv2.imwrite(os.path.join(pasta_baseline, nome_arquivo), img)
        
        landmarks = resultados.multi_face_landmarks[0].landmark
        img_hmd = img.copy()
        img_hmd_olhos = img.copy()

        # Matemática da Ancoragem Rígida (Crânio)
        pt_esq, pt_dir = landmarks[234], landmarks[454]
        pt_centro = landmarks[168]
        
        x_esq, y_esq = int(pt_esq.x * w), int(pt_esq.y * h)
        x_dir, y_dir = int(pt_dir.x * w), int(pt_dir.y * h)
        x_c, y_c = int(pt_centro.x * w), int(pt_centro.y * h)

        dist_cranio = math.hypot(x_dir - x_esq, y_dir - y_esq)
        largura = int(dist_cranio * 1.30)
        altura = int(dist_cranio * 0.70)
        angulo_rad = math.atan2(y_dir - y_esq, x_dir - x_esq)
        angulo_graus = math.degrees(angulo_rad)

        retangulo = ((x_c, y_c), (largura, altura), angulo_graus)
        box = np.intp(cv2.boxPoints(retangulo))
        
        COR_HMD = (0, 0, 0)
        cv2.fillPoly(img_hmd, [box], COR_HMD)
        cv2.fillPoly(img_hmd_olhos, [box], COR_HMD)

        px_esq = (int(landmarks[468].x * w), int(landmarks[468].y * h))
        px_dir = (int(landmarks[473].x * w), int(landmarks[473].y * h))
        
        eixo_x = int(largura * 0.08)
        eixo_y = int(largura * 0.035)
        raio_iris = int(eixo_y * 0.8)
        
        cv2.ellipse(img_hmd_olhos, px_esq, (eixo_x, eixo_y), angulo_graus, 0, 360, (255, 255, 255), -1)
        cv2.circle(img_hmd_olhos, px_esq, raio_iris, (0, 0, 0), -1)
        
        cv2.ellipse(img_hmd_olhos, px_dir, (eixo_x, eixo_y), angulo_graus, 0, 360, (255, 255, 255), -1)
        cv2.circle(img_hmd_olhos, px_dir, raio_iris, (0, 0, 0), -1)

        cv2.imwrite(os.path.join(pasta_hmd, nome_arquivo), img_hmd)
        cv2.imwrite(os.path.join(pasta_hmd_olhos, nome_arquivo), img_hmd_olhos)
        return True
        
    else:
        # --- FALHA: Adiciona a tag '_error' ao nome do arquivo ---
        nome_base, extensao = os.path.splitext(nome_arquivo)
        nome_com_erro = f"{nome_base}_error{extensao}"
        
        # Salva a imagem original nas 3 pastas para manter a contagem, mas demarcada com erro
        cv2.imwrite(os.path.join(pasta_baseline, nome_com_erro), img)
        cv2.imwrite(os.path.join(pasta_hmd, nome_com_erro), img)
        cv2.imwrite(os.path.join(pasta_hmd_olhos, nome_com_erro), img)
        return False

def rodar_extracao_rafdb():
    PASTA_IMAGENS_ORIGINAIS = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\dataset\RAF-DB\DATASET" 
    ARQUIVO_TRAIN_CSV = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\dataset\RAF-DB\train_labels.csv"
    ARQUIVO_TEST_CSV = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\dataset\RAF-DB\test_labels.csv"
    
    PASTA_DESTINO_RAIZ = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\dataset_processado\RAF-DB"
    
    splits = {
        'train': ARQUIVO_TRAIN_CSV,
        'test': ARQUIVO_TEST_CSV
    }

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5
    ) as face_mesh:

        for split_name, csv_path in splits.items():
            if not os.path.exists(csv_path):
                continue
            
            print(f"\n[INFO] Processando lote: {split_name.upper()}")
            
            try:
                df = pd.read_csv(csv_path)
                if len(df.columns) == 1:
                    df = pd.read_csv(csv_path, sep=' ', header=None, names=['image', 'label'])
            except Exception as e:
                continue

            falhas_deteccao = 0
            imagens_nao_encontradas = 0
            
            for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processando {split_name}"):
                nome_imagem = row['image']
                label_num = int(row['label'])
                nome_emocao = EMOCOES_RAF.get(label_num, '8_Unknown')

                caminho_origem = os.path.join(PASTA_IMAGENS_ORIGINAIS, split_name, str(label_num), nome_imagem)
                
                if not os.path.exists(caminho_origem):
                    imagens_nao_encontradas += 1
                    continue
                
                pasta_base = os.path.join(PASTA_DESTINO_RAIZ, "Baseline", split_name, nome_emocao)
                pasta_hmd = os.path.join(PASTA_DESTINO_RAIZ, "HMD", split_name, nome_emocao)
                pasta_olhos = os.path.join(PASTA_DESTINO_RAIZ, "HMD_Olhos", split_name, nome_emocao)

                os.makedirs(pasta_base, exist_ok=True)
                os.makedirs(pasta_hmd, exist_ok=True)
                os.makedirs(pasta_olhos, exist_ok=True)

                sucesso = processar_imagem_rafdb(
                    caminho_origem, pasta_base, pasta_hmd, pasta_olhos, nome_imagem, face_mesh
                )
                
                if not sucesso:
                    falhas_deteccao += 1
            
            print(f"-> Concluído {split_name}!")
            print(f"   - Imagens não encontradas no disco: {imagens_nao_encontradas}")
            print(f"   - Falhas de detecção (Marcadas com _error): {falhas_deteccao}")

if __name__ == "__main__":
    rodar_extracao_rafdb()
    print("\n[SUCESSO] Pipeline do RAF-DB finalizado!")