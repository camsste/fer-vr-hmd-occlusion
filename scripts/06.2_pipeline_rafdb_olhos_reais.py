import cv2
import mediapipe as mp
import numpy as np
import os
import math
import pandas as pd
from tqdm import tqdm

# Dicionário de emoções do RAF-DB
EMOCOES_RAF = {
    1: '1_Surprise', 2: '2_Fear', 3: '3_Disgust', 4: '4_Happiness',
    5: '5_Sadness', 6: '6_Anger', 7: '7_Neutral'
}

mp_face_mesh = mp.solutions.face_mesh

def sobrepor_imagem_transparente(fundo, sobreposicao, x_centro, y_centro, largura_desejada):
    # --- TRAVA DE SEGURANÇA ---
    if largura_desejada <= 0:
        return fundo
    
    h_orig, w_orig = sobreposicao.shape[:2]
    proporcao = largura_desejada / float(w_orig)
    altura_desejada = int(h_orig * proporcao)
    
    if altura_desejada <= 0:
        return fundo
    # --------------------------
    
    sobreposicao_redimensionada = cv2.resize(sobreposicao, (largura_desejada, altura_desejada), interpolation=cv2.INTER_AREA)
    
    x = int(x_centro - (largura_desejada / 2))
    y = int(y_centro - (altura_desejada / 2))
    
    h_sobre, w_sobre = sobreposicao_redimensionada.shape[:2]
    h_fundo, w_fundo = fundo.shape[:2]

    # Prevenção contra colagem fora das bordas (agora mais robusta)
    if x < 0 or y < 0 or x + w_sobre > w_fundo or y + h_sobre > h_fundo:
        return fundo

    # Separa os canais (esperando BGRA)
    b, g, r, a = cv2.split(sobreposicao_redimensionada)
    overlay_rgb = cv2.merge((b, g, r))

    mask = a / 255.0
    mask_inv = 1.0 - mask

    roi = fundo[y:y+h_sobre, x:x+w_sobre]

    for c in range(0, 3):
        roi[:, :, c] = (mask * overlay_rgb[:, :, c] + mask_inv * roi[:, :, c])

    fundo[y:y+h_sobre, x:x+w_sobre] = roi
    return fundo

def processar_imagem_reais(caminho_img_origem, pasta_destino, nome_arquivo, face_mesh, img_olho_esq, img_olho_dir):
    img = cv2.imread(caminho_img_origem)
    if img is None:
        return False

    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultados = face_mesh.process(img_rgb)
    
    if resultados.multi_face_landmarks:
        landmarks = resultados.multi_face_landmarks[0].landmark
        img_final = img.copy()

        # 1. Desenha a Tarja Preta (HMD) - Mesma matemática anterior
        pt_esq, pt_dir = landmarks[234], landmarks[454]
        pt_centro = landmarks[168]
        
        x_esq, y_esq = int(pt_esq.x * w), int(pt_esq.y * h)
        x_dir, y_dir = int(pt_dir.x * w), int(pt_dir.y * h)
        x_c, y_c = int(pt_centro.x * w), int(pt_centro.y * h)

        dist_cranio = math.hypot(x_dir - x_esq, y_dir - y_esq)
        largura_hmd = int(dist_cranio * 1.30)
        altura_hmd = int(dist_cranio * 0.70)
        angulo_graus = math.degrees(math.atan2(y_dir - y_esq, x_dir - x_esq))

        retangulo = ((x_c, y_c), (largura_hmd, altura_hmd), angulo_graus)
        box = np.intp(cv2.boxPoints(retangulo))
        cv2.fillPoly(img_final, [box], (0, 0, 0))

        # 2. Cola os PNGs Realistas
        # Calculamos a largura do olho baseada na distância entre os cantos (ex: landmarks 33 e 133 para o esquerdo)
        dist_olho_esq = math.hypot((landmarks[133].x - landmarks[33].x) * w, (landmarks[133].y - landmarks[33].y) * h)
        dist_olho_dir = math.hypot((landmarks[263].x - landmarks[362].x) * w, (landmarks[263].y - landmarks[362].y) * h)
        
        # Ampliamos a largura calculada em 30% para garantir que cubra bem a área
        largura_png_esq = int(dist_olho_esq * 1.3)
        largura_png_dir = int(dist_olho_dir * 1.3)

        # Coordenadas das pupilas
        px_esq = int(landmarks[468].x * w)
        py_esq = int(landmarks[468].y * h)
        px_dir = int(landmarks[473].x * w)
        py_dir = int(landmarks[473].y * h)

        # Aplica o olho esquerdo
        img_final = sobrepor_imagem_transparente(img_final, img_olho_esq, px_esq, py_esq, largura_png_esq)
        # Aplica o olho direito
        img_final = sobrepor_imagem_transparente(img_final, img_olho_dir, px_dir, py_dir, largura_png_dir)

        cv2.imwrite(os.path.join(pasta_destino, nome_arquivo), img_final)
        return True
    else:
        # Se falhar, salva com tag de erro
        nome_base, extensao = os.path.splitext(nome_arquivo)
        cv2.imwrite(os.path.join(pasta_destino, f"{nome_base}_error{extensao}"), img)
        return False

def rodar_extracao_olhos_reais():
    # --- CAMINHOS DAS IMAGENS ---
    PASTA_IMAGENS_ORIGINAIS = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\dataset\RAF-DB\DATASET" 
    PASTA_DESTINO_RAIZ = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\dataset_processado\RAF-DB"
    
    # ATENÇÃO: Ajuste estes caminhos para onde você baixou os PNGs que você me enviou
    CAMINHO_PNG_ESQ = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\Olho_Esquerdo.png"
    CAMINHO_PNG_DIR = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\Olho_Direito.png"

    ARQUIVO_TRAIN_CSV = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\dataset\RAF-DB\train_labels.csv"
    ARQUIVO_TEST_CSV = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\dataset\RAF-DB\test_labels.csv"

    # Carrega os PNGs mantendo o canal alfa (IMREAD_UNCHANGED)
    olho_esq = cv2.imread(CAMINHO_PNG_ESQ, cv2.IMREAD_UNCHANGED)
    olho_dir = cv2.imread(CAMINHO_PNG_DIR, cv2.IMREAD_UNCHANGED)

    if olho_esq is None or olho_dir is None:
        print("[ERRO] Arquivos PNG dos olhos não encontrados! Verifique os caminhos.")
        return

    splits = {'train': ARQUIVO_TRAIN_CSV, 'test': ARQUIVO_TEST_CSV}

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        for split_name, csv_path in splits.items():
            if not os.path.exists(csv_path): continue
            
            print(f"\n[INFO] Gerando cenário HMD_Olhos_Reais para: {split_name.upper()}")
            try:
                df = pd.read_csv(csv_path)
                if len(df.columns) == 1:
                    df = pd.read_csv(csv_path, sep=' ', header=None, names=['image', 'label'])
            except Exception as e:
                continue

            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                nome_imagem = row['image']
                label_num = int(row['label'])
                nome_emocao = EMOCOES_RAF.get(label_num, '8_Unknown')

                caminho_origem = os.path.join(PASTA_IMAGENS_ORIGINAIS, split_name, str(label_num), nome_imagem)
                if not os.path.exists(caminho_origem): continue
                
                # Cria a nova pasta específica para esta abordagem
                pasta_destino = os.path.join(PASTA_DESTINO_RAIZ, "HMD_Olhos_Reais", split_name, nome_emocao)
                os.makedirs(pasta_destino, exist_ok=True)

                processar_imagem_reais(caminho_origem, pasta_destino, nome_imagem, face_mesh, olho_esq, olho_dir)

if __name__ == "__main__":
    rodar_extracao_olhos_reais()
    print("\n[SUCESSO] Nova pasta HMD_Olhos_Reais gerada!")