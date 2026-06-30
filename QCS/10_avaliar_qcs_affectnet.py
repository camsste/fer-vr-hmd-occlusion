# --- 1. BLINDAGEM DE COMPATIBILIDADE PYTORCH ---
import sys
import types
import collections.abc
import torch
import os

torch_six = types.ModuleType('torch._six')
torch_six.container_abcs = collections.abc
sys.modules['torch._six'] = torch_six

# --- CLASSES FANTASMAS (Burlar arquivos do Checkpoint) ---
class RecorderMeter_matrix(object): pass
class RecorderMeter(object): pass
class RecorderMeter1(object): pass
class RecorderMeter_loss(object): pass
class AverageMeter(object): pass
class ProgressMeter(object): pass
# ---------------------------------------------------------

# --- 2. BLINDAGEM DO BUG DE HARDCODE DO QCS ---
_original_load = torch.load
def safe_load(f, *args, **kwargs):
    if isinstance(f, str) and ('pretrain' in f or 'QCS_affect' in f or 'ir50' in f):
        return {} 
    return _original_load(f, *args, **kwargs)
torch.load = safe_load

DIRETORIO_QCS = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\QCS"
sys.path.append(DIRETORIO_QCS)
# ------------------------------------------------

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm

# Importa a arquitetura específica do AffectNet diretamente do repositório QCS
from models.QCS_7cls_affectnet import pyramid_trans_expr

def avaliar_cenario(modelo, caminho_pasta, device):
    nome_cenario = os.path.basename(os.path.dirname(caminho_pasta))
    print(f"\n{'='*50}")
    print(f"--- AVALIANDO CENÁRIO QCS (AFFECTNET): {nome_cenario.upper()} ---")
    print(f"{'='*50}")
    
    transformacoes = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        # AffectNet utiliza a pasta 'val'
        dataset = datasets.ImageFolder(root=caminho_pasta, transform=transformacoes)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0) 
    except Exception as e:
        print(f"[ERRO] Não foi possível carregar as imagens em {caminho_pasta}: {e}")
        return

    classes = dataset.classes
    todas_preds = []
    todas_labels = []

    print(f"[INFO] Classes identificadas: {classes}")
    
    with torch.no_grad():
        # Dicionário de tradução: Nó do Modelo -> Índice da Pasta PyTorch
        mapa_correcao = {
            0: 6, # Modelo diz Neutral (0) -> Pasta é 7_Neutral (índice 6)
            1: 3, # Modelo diz Happy (1) -> Pasta é 4_Happiness (índice 3)
            2: 4, # Modelo diz Sad (2) -> Pasta é 5_Sadness (índice 4)
            3: 0, # Modelo diz Surprise (3) -> Pasta é 1_Surprise (índice 0)
            4: 1, # Modelo diz Fear (4) -> Pasta é 2_Fear (índice 1)
            5: 2, # Modelo diz Disgust (5) -> Pasta é 3_Disgust (índice 2)
            6: 5  # Modelo diz Anger (6) -> Pasta é 6_Anger (índice 5)
        }

        for imagens, labels in tqdm(dataloader, desc=f"Inferência ({nome_cenario})", unit="lote"):
            imagens = imagens.to(device)
            labels = labels.to(device)

            # Passa a imagem Anchor e desativa os ramos cruzados
            saida = modelo(imagens, None, None, None)
            
            _, preds = torch.max(saida, 1)
            
            # Traduz a predição crua do modelo para o índice alfabético da pasta
            preds_corrigidas = [mapa_correcao[p.item()] for p in preds]
            
            todas_preds.extend(preds_corrigidas)
            todas_labels.extend(labels.cpu().numpy())

    if len(todas_preds) == 0:
        return

    acc = accuracy_score(todas_labels, todas_preds)
    f1_macro = f1_score(todas_labels, todas_preds, average='macro')
    matriz_conf = confusion_matrix(todas_labels, todas_preds)
    
    nomes_presentes = [classes[i] for i in np.unique(todas_labels)]
    relatorio = classification_report(todas_labels, todas_preds, target_names=nomes_presentes, zero_division=0)

    print(f"\n=> RESULTADOS QCS AFFECTNET ({nome_cenario}):")
    print(f"Acurácia Global: {acc*100:.2f}%")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print("\n=> MATRIZ DE CONFUSÃO:")
    print(matriz_conf)
    print("\n=> RELATÓRIO:")
    print(relatorio)

def executar_qcs_affectnet():
    device = torch.device("cpu")
    print(f"\n[SISTEMA] Iniciando avaliação do QCS (AffectNet) via: {str(device).upper()}")

    print("[SISTEMA] Montando a arquitetura QCS (pyramid_trans_expr)...")
    modelo = pyramid_trans_expr(img_size=224, num_classes=7)
    
    # Aponta para o novo arquivo baixado
    caminho_pesos = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\QCS\checkpoint\qcs_affectnet7.pth"
    
    if not os.path.exists(caminho_pesos):
        print(f"[ERRO FATAL] Arquivo de pesos não encontrado em: {caminho_pesos}")
        return

    print(f"[SISTEMA] Injetando a inteligência artificial do arquivo: {caminho_pesos}...")
    checkpoint = _original_load(caminho_pesos, map_location=device, weights_only=False)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    novo_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            novo_state_dict[key[7:]] = value
        else:
            novo_state_dict[key] = value

    modelo.load_state_dict(novo_state_dict, strict=False)
    modelo.to(device)
    modelo.eval()

    # Caminho base do AffectNet
    base_dir = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\dataset_processado\AffectNet"
    
    # As 4 pastas de validação do AffectNet
    cenarios = [
        os.path.join(base_dir, "Baseline", "val"),
        os.path.join(base_dir, "HMD", "val"),
        os.path.join(base_dir, "HMD_Olhos", "val"),
        os.path.join(base_dir, "HMD_Olhos_Reais", "val")
    ]

    for cenario in cenarios:
        if os.path.exists(cenario):
            avaliar_cenario(modelo, cenario, device)
        else:
            print(f"\n[AVISO] A pasta {cenario} não existe!")

if __name__ == "__main__":
    executar_qcs_affectnet()