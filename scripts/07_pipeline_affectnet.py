import cv2
import mediapipe as mp
import numpy as np
import os
import math
from tqdm import tqdm

# Mapeamento oficial de pastas para labels no AffectNet (7 classes)
MAPA_EMOCOES = {
    '0': '0_Neutral', '1': '1_Happiness', '2': '2_Sadness',
    '3': '3_Surprise', '4': '4_Fear', '5': '5_Disgust', '6': '6_Anger'
}

mp_face_mesh = mp.solutions.face_mesh

def rotacionar_png(img, angulo, centro):
    """Rotaciona o PNG mantendo a transparência (Alpha)."""
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, 
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated

def sobrepor_imagem_transparente(fundo, sobreposicao, x_centro, y_centro, largura_desejada):
    """Cola PNG transparente sobre o fundo, com redimensionamento inteligente."""
    if largura_desejada <= 0: return fundo
    
    h_orig, w_orig = sobreposicao.shape[:2]
    proporcao = largura_desejada / float(w_orig)
    altura_desejada = int(h_orig * proporcao)
    
    if altura_desejada <= 0: return fundo
    
    img_rz = cv2.resize(sobreposicao, (largura_desejada, altura_desejada), interpolation=cv2.INTER_AREA)
    
    x = int(x_centro - (largura_desejada / 2))
    y = int(y_centro - (altura_desejada / 2))
    
    if x < 0 or y < 0 or x + img_rz.shape[1] > fundo.shape[1] or y + img_rz.shape[0] > fundo.shape[0]:
        return fundo

    b, g, r, a = cv2.split(img_rz)
    mask = a / 255.0
    
    for c in range(0, 3):
        roi = fundo[y:y+img_rz.shape[0], x:x+img_rz.shape[1], c]
        fundo[y:y+img_rz.shape[0], x:x+img_rz.shape[1], c] = (mask * img_rz[:,:,c] + (1.0 - mask) * roi)
    
    return fundo

def processar_imagem(caminho_img_origem, dirs_destino, nome_arquivo, face_mesh, olho_esq, olho_dir):
    img = cv2.imread(caminho_img_origem)
    if img is None: return False

    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(img_rgb)
    
    if res.multi_face_landmarks:
        l = res.multi_face_landmarks[0].landmark
        
        b_img = img.copy()
        h_img = img.copy()
        o_img = img.copy()
        r_img = img.copy()

        # --- 1. Matemática da Ancoragem Rígida do HMD ---
        pt_esq, pt_dir, pt_centro = l[234], l[454], l[168]
        x_esq, y_esq = int(pt_esq.x * w), int(pt_esq.y * h)
        x_dir, y_dir = int(pt_dir.x * w), int(pt_dir.y * h)
        
        dist_cranio = math.hypot(x_dir - x_esq, y_dir - y_esq)
        largura_hmd = int(dist_cranio * 1.30)
        altura_hmd = int(dist_cranio * 0.70)
        angulo_graus = math.degrees(math.atan2(y_dir - y_esq, x_dir - x_esq))
        
        ret = ((int(pt_centro.x * w), int(pt_centro.y * h)), (largura_hmd, altura_hmd), angulo_graus)
        box = np.intp(cv2.boxPoints(ret))
        
        # Desenha a tarja preta nos três cenários
        cv2.fillPoly(h_img, [box], (0, 0, 0))
        cv2.fillPoly(o_img, [box], (0, 0, 0))
        cv2.fillPoly(r_img, [box], (0, 0, 0))
        
        # Coordenadas das pupilas
        px_esq = (int(l[468].x * w), int(l[468].y * h))
        px_dir = (int(l[473].x * w), int(l[473].y * h))
        
        # --- 2. Olhos Artificiais (Recuperando a qualidade original: elipse branca + íris preta) ---
        eixo_x = int(largura_hmd * 0.08)
        eixo_y = int(largura_hmd * 0.035)
        raio_iris = int(eixo_y * 0.8)
        
        # Olho Esquerdo Artificial
        cv2.ellipse(o_img, px_esq, (eixo_x, eixo_y), angulo_graus, 0, 360, (255, 255, 255), -1)
        cv2.circle(o_img, px_esq, raio_iris, (0, 0, 0), -1)
        # Olho Direito Artificial
        cv2.ellipse(o_img, px_dir, (eixo_x, eixo_y), angulo_graus, 0, 360, (255, 255, 255), -1)
        cv2.circle(o_img, px_dir, raio_iris, (0, 0, 0), -1)
        
        # --- 3. Olhos Reais (Matemática correta da distância anatômica dos olhos + Rotação) ---
        dist_olho_esq = math.hypot((l[133].x - l[33].x) * w, (l[133].y - l[33].y) * h)
        dist_olho_dir = math.hypot((l[263].x - l[362].x) * w, (l[263].y - l[362].y) * h)
        
        largura_png_esq = int(dist_olho_esq * 1.3)
        largura_png_dir = int(dist_olho_dir * 1.3)
        
        png_esq_rot = rotacionar_png(olho_esq, angulo_graus, (olho_esq.shape[1]/2, olho_esq.shape[0]/2))
        png_dir_rot = rotacionar_png(olho_dir, angulo_graus, (olho_dir.shape[1]/2, olho_dir.shape[0]/2))
        
        r_img = sobrepor_imagem_transparente(r_img, png_esq_rot, px_esq[0], px_esq[1], largura_png_esq)
        r_img = sobrepor_imagem_transparente(r_img, png_dir_rot, px_dir[0], px_dir[1], largura_png_dir)

        # --- Salvar os 4 cenários ---
        cv2.imwrite(os.path.join(dirs_destino["Baseline"], nome_arquivo), b_img)
        cv2.imwrite(os.path.join(dirs_destino["HMD"], nome_arquivo), h_img)
        cv2.imwrite(os.path.join(dirs_destino["HMD_Olhos"], nome_arquivo), o_img)
        cv2.imwrite(os.path.join(dirs_destino["HMD_Olhos_Reais"], nome_arquivo), r_img)
        return True
    else:
        # Fallback de erro: Salva a imagem intocada mas demarca com "error"
        nome_err = nome_arquivo.replace(".", "_error.")
        for pasta in dirs_destino.values():
            cv2.imwrite(os.path.join(pasta, nome_err), img)
        return False

def executar_pipeline():
    RAIZ_DATASET = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\dataset\affectnet"
    RAIZ_DESTINO = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\dataset_processado\AffectNet"
    
    # ATENÇÃO aos caminhos dos PNGs
    olho_esq = cv2.imread(r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\Olho_Esquerdo.png", cv2.IMREAD_UNCHANGED)
    olho_dir = cv2.imread(r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\Olho_Direito.png", cv2.IMREAD_UNCHANGED)

    if olho_esq is None or olho_dir is None:
        print("[ERRO] Arquivos PNG dos olhos não encontrados!")
        return

    # O AffectNet exige leitura de pastas (ignorando CSV)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        for fase in ['train', 'val']:
            caminho_fase = os.path.join(RAIZ_DATASET, fase)
            if not os.path.exists(caminho_fase): continue

            for pasta_label in os.listdir(caminho_fase):
                if pasta_label not in MAPA_EMOCOES: continue
                
                nome_emocao = MAPA_EMOCOES[pasta_label]
                caminho_origem = os.path.join(caminho_fase, pasta_label)
                
                # Prepara pastas de saída completas
                dirs_destino = {
                    "Baseline": os.path.join(RAIZ_DESTINO, "Baseline", fase, nome_emocao),
                    "HMD": os.path.join(RAIZ_DESTINO, "HMD", fase, nome_emocao),
                    "HMD_Olhos": os.path.join(RAIZ_DESTINO, "HMD_Olhos", fase, nome_emocao),
                    "HMD_Olhos_Reais": os.path.join(RAIZ_DESTINO, "HMD_Olhos_Reais", fase, nome_emocao)
                }
                for d in dirs_destino.values(): os.makedirs(d, exist_ok=True)

                arquivos = [f for f in os.listdir(caminho_origem) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                
                for arquivo in tqdm(arquivos, desc=f"Processando {fase}/{nome_emocao}"):
                    caminho_img = os.path.join(caminho_origem, arquivo)
                    processar_imagem(caminho_img, dirs_destino, arquivo, face_mesh, olho_esq, olho_dir)

if __name__ == "__main__":
    executar_pipeline()
    print("\n[SUCESSO] Pipeline completo e fiel concluído!")