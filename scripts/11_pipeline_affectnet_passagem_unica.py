import cv2
import mediapipe as mp
import numpy as np
import os
import json
import math
from tqdm import tqdm

# ==============================================================================
# 1. CONFIGURAÇÕES E CAMINHOS ATUALIZADOS (AFFECTNET+ OFICIAL)
# ==============================================================================
ORIGEM_BASE = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\dataset\AffectNet+\human_annotated\validation_set"
ORIGEM_JSON = os.path.join(ORIGEM_BASE, "annotations")
ORIGEM_IMAGES = os.path.join(ORIGEM_BASE, "images")

DESTINO_BASE = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\dataset_processado\AffectNet"

CAMINHO_OLHO_ESQ = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\Olho_Esquerdo.png"
CAMINHO_OLHO_DIR = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\Olho_Direito.png"

MAPA_EMOCOES = {
    0: '7_Neutral', 1: '4_Happiness', 2: '5_Sadness',
    3: '1_Surprise', 4: '2_Fear', 5: '3_Disgust', 6: '6_Anger'
}

LIMITE_POR_CLASSE = 500
contadores = {nome: 0 for nome in MAPA_EMOCOES.values()}

# ==============================================================================
# 2. MATEMÁTICA VISUAL (COMPOSIÇÃO E ROTAÇÃO)
# ==============================================================================
def rotacionar_png(img, angulo, centro):
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, 
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated

def sobrepor_imagem_transparente(fundo, sobreposicao, x_centro, y_centro, largura_desejada):
    if largura_desejada <= 0: return fundo
    h_orig, w_orig = sobreposicao.shape[:2]
    proporcao = largura_desejada / float(w_orig)
    altura_desejada = int(h_orig * proporcao)
    if altura_desejada <= 0: return fundo
    img_rz = cv2.resize(sobreposicao, (largura_desejada, altura_desejada), interpolation=cv2.INTER_AREA)
    x = int(x_centro - (largura_desejada / 2))
    y = int(y_centro - (altura_desejada / 2))
    if x < 0 or y < 0 or x + img_rz.shape[1] > fundo.shape[1] or y + img_rz.shape[0] > fundo.shape[0]: return fundo
    b, g, r, a = cv2.split(img_rz)
    mask = a / 255.0
    for c in range(0, 3):
        roi = fundo[y:y+img_rz.shape[0], x:x+img_rz.shape[1], c]
        fundo[y:y+img_rz.shape[0], x:x+img_rz.shape[1], c] = (mask * img_rz[:,:,c] + (1.0 - mask) * roi)
    return fundo

# ==============================================================================
# 3. SUPER PIPELINE (COM FALLBACK PARA IMAGENS DIFÍCEIS)
# ==============================================================================
def processar_dataset():
    print("[SISTEMA] Iniciando a filtragem e geração de cenários (AffectNet+)")

    olho_esq_img = cv2.imread(CAMINHO_OLHO_ESQ, cv2.IMREAD_UNCHANGED)
    olho_dir_img = cv2.imread(CAMINHO_OLHO_DIR, cv2.IMREAD_UNCHANGED)
    
    if olho_esq_img is None or olho_dir_img is None:
        print(f"[ERRO FATAL] Imagens PNG não encontradas!\nEsq: {CAMINHO_OLHO_ESQ}\nDir: {CAMINHO_OLHO_DIR}")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

    fases = ["Baseline", "HMD", "HMD_Olhos", "HMD_Olhos_Reais"]
    for fase in fases:
        for classe in MAPA_EMOCOES.values():
            os.makedirs(os.path.join(DESTINO_BASE, fase, "val", classe), exist_ok=True)

    arquivos_json = [f for f in os.listdir(ORIGEM_JSON) if f.endswith('.json')]
    pbar = tqdm(total=LIMITE_POR_CLASSE * len(MAPA_EMOCOES), desc="Processando Imagens")

    for arquivo in arquivos_json:
        if all(c >= LIMITE_POR_CLASSE for c in contadores.values()):
            break

        caminho_json = os.path.join(ORIGEM_JSON, arquivo)
        with open(caminho_json, 'r') as f:
            dados = json.load(f)
            
        label_id = dados.get("human-label")
        if label_id not in MAPA_EMOCOES:
            continue
            
        nome_emocao = MAPA_EMOCOES[label_id]
        if contadores[nome_emocao] >= LIMITE_POR_CLASSE:
            continue

        nome_imagem = arquivo.replace('.json', '.jpg')
        caminho_img_origem = os.path.join(ORIGEM_IMAGES, nome_imagem)
        
        if not os.path.exists(caminho_img_origem):
            print(f"\n[FALHA] Imagem ausente no diretório: {nome_imagem}")
            continue

        img = cv2.imread(caminho_img_origem)
        if img is None:
            print(f"\n[FALHA] Arquivo corrompido: {nome_imagem}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        # Prepara as cópias originais para caso de Fallback
        b_img = img.copy()
        h_img = img.copy()
        o_img = img.copy()
        r_img = img.copy()

        # O Filtro: Se o MediaPipe encontrar o rosto, aplicamos as tarjas e olhos
        # Se NÃO encontrar, a imagem segue intacta para os 4 cenários (Fallback)
        if not results.multi_face_landmarks:
            print(f"\n[AVISO - FALLBACK] Rosto não detectado em {nome_imagem} ({nome_emocao}). Salvando imagem sem oclusão.")
        else:
            h, w, _ = img.shape
            l = results.multi_face_landmarks[0].landmark

            pt_esq, pt_dir, pt_centro = l[234], l[454], l[168]
            x_esq, y_esq = int(pt_esq.x * w), int(pt_esq.y * h)
            x_dir, y_dir = int(pt_dir.x * w), int(pt_dir.y * h)
            
            dist_cranio = math.hypot(x_dir - x_esq, y_dir - y_esq)
            largura_hmd = int(dist_cranio * 1.30)
            altura_hmd = int(dist_cranio * 0.70)
            angulo_graus = math.degrees(math.atan2(y_dir - y_esq, x_dir - x_esq))
            
            ret = ((int(pt_centro.x * w), int(pt_centro.y * h)), (largura_hmd, altura_hmd), angulo_graus)
            box = np.intp(cv2.boxPoints(ret))
            
            px_esq = (int(l[468].x * w), int(l[468].y * h))
            px_dir = (int(l[473].x * w), int(l[473].y * h))
            
            cv2.fillPoly(h_img, [box], (0, 0, 0))

            o_img = h_img.copy()
            eixo_x = int(largura_hmd * 0.08)
            eixo_y = int(largura_hmd * 0.035)
            raio_iris = int(eixo_y * 0.8)
            
            cv2.ellipse(o_img, px_esq, (eixo_x, eixo_y), angulo_graus, 0, 360, (255, 255, 255), -1)
            cv2.circle(o_img, px_esq, raio_iris, (0, 0, 0), -1)
            cv2.ellipse(o_img, px_dir, (eixo_x, eixo_y), angulo_graus, 0, 360, (255, 255, 255), -1)
            cv2.circle(o_img, px_dir, raio_iris, (0, 0, 0), -1)

            r_img = h_img.copy()
            dist_olho_esq = math.hypot((l[133].x - l[33].x) * w, (l[133].y - l[33].y) * h)
            dist_olho_dir = math.hypot((l[263].x - l[362].x) * w, (l[263].y - l[362].y) * h)
            
            largura_png_esq = int(dist_olho_esq * 1.3)
            largura_png_dir = int(dist_olho_dir * 1.3)
            
            png_esq_rot = rotacionar_png(olho_esq_img, angulo_graus, (olho_esq_img.shape[1]/2, olho_esq_img.shape[0]/2))
            png_dir_rot = rotacionar_png(olho_dir_img, angulo_graus, (olho_dir_img.shape[1]/2, olho_dir_img.shape[0]/2))
            
            r_img = sobrepor_imagem_transparente(r_img, png_esq_rot, px_esq[0], px_esq[1], largura_png_esq)
            r_img = sobrepor_imagem_transparente(r_img, png_dir_rot, px_dir[0], px_dir[1], largura_png_dir)

        # Salva o resultado (Seja processado ou o fallback intocado)
        cv2.imwrite(os.path.join(DESTINO_BASE, "Baseline", "val", nome_emocao, nome_imagem), b_img)
        cv2.imwrite(os.path.join(DESTINO_BASE, "HMD", "val", nome_emocao, nome_imagem), h_img)
        cv2.imwrite(os.path.join(DESTINO_BASE, "HMD_Olhos", "val", nome_emocao, nome_imagem), o_img)
        cv2.imwrite(os.path.join(DESTINO_BASE, "HMD_Olhos_Reais", "val", nome_emocao, nome_imagem), r_img)

        # Computa a imagem para a classe e avança
        contadores[nome_emocao] += 1
        pbar.update(1)

    pbar.close()
    face_mesh.close()

    print("\n" + "="*40)
    print("RELATÓRIO FINAL DO DATASET GERADO")
    print("="*40)
    total_gerado = 0
    for emocao, contagem in contadores.items():
        print(f"{emocao}: {contagem}/500")
        total_gerado += contagem
    print(f"TOTAL: {total_gerado} imagens processadas com sucesso.")

if __name__ == "__main__":
    processar_dataset()