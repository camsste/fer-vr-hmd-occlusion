# --- PATCH DE COMPATIBILIDADE PARA PYTORCH MODERNO ---
import sys
import types
import collections.abc
import torch
torch_six = types.ModuleType('torch._six')
torch_six.container_abcs = collections.abc
sys.modules['torch._six'] = torch_six
# -----------------------------------------------------

# --- CLASSES FANTASMAS (Para evitar erros do POSTER++) ---
class RecorderMeter(object): pass
class RecorderMeter1(object): pass
# ---------------------------------------------------------

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm

# Importação da arquitetura oficial do POSTER++
from models.PosterV2_7cls import pyramid_trans_expr2

def avaliar_cenario(modelo, caminho_pasta, device):
    nome_cenario = os.path.basename(os.path.dirname(caminho_pasta))
    print(f"\n{'='*50}")
    print(f"--- AVALIANDO CENÁRIO: {nome_cenario.upper()} ---")
    print(f"{'='*50}")
    
    transformacoes = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        # Lê a pasta de validação ('val')
        dataset = datasets.ImageFolder(root=caminho_pasta, transform=transformacoes)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0) 
    except Exception as e:
        print(f"[ERRO] Não foi possível carregar as imagens em {caminho_pasta}: {e}")
        return

    classes = dataset.classes
    todas_preds = []
    todas_labels = []

    print(f"[INFO] Classes identificadas: {classes}")
    print(f"[INFO] Lendo imagens de {caminho_pasta}...")
    
    with torch.no_grad():
        # Dicionário de tradução: Nó do Modelo -> Índice da Pasta
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

            saida = modelo(imagens)
            
            if isinstance(saida, tuple):
                saida = saida[0]
                
            _, preds = torch.max(saida, 1)
            
            # Traduz a predição crua do modelo para o índice alfabético da pasta
            preds_corrigidas = [mapa_correcao[p.item()] for p in preds]
            
            todas_preds.extend(preds_corrigidas)
            todas_labels.extend(labels.cpu().numpy())

    if len(todas_preds) == 0:
        print("[AVISO] Nenhuma predição feita. A pasta está vazia?")
        return

    acc = accuracy_score(todas_labels, todas_preds)
    f1_macro = f1_score(todas_labels, todas_preds, average='macro')
    matriz_conf = confusion_matrix(todas_labels, todas_preds)
    
    labels_presentes = np.unique(todas_labels)
    nomes_presentes = [classes[i] for i in labels_presentes]
    relatorio = classification_report(todas_labels, todas_preds, target_names=nomes_presentes, zero_division=0)

    print(f"\n=> RESULTADOS CONSOLIDADOS:")
    print(f"Acurácia Global: {acc*100:.2f}%")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    
    print("\n=> MATRIZ DE CONFUSÃO:")
    print(matriz_conf)
    
    print("\n=> RELATÓRIO POR EMOÇÃO (F1, Precision, Recall):")
    print(relatorio)

def executar_pipeline():
    # Usando a CPU por segurança de compatibilidade
    device = torch.device("cpu")
    print(f"\n[SISTEMA] Iniciando processamento via: {str(device).upper()} (Modo de Compatibilidade)")

    print("[SISTEMA] Montando a arquitetura POSTER_V2 (pyramid_trans_expr2)...")
    modelo = pyramid_trans_expr2(img_size=224, num_classes=7)
    
    # ATENÇÃO: Ajuste o nome do arquivo para o peso do AffectNet que você baixou
    caminho_pesos = r"./checkpoint/affectnet-7-model_best.pth" 
    
    if not os.path.exists(caminho_pesos):
        print(f"[ERRO FATAL] O arquivo de pesos não foi encontrado em: {caminho_pesos}")
        return

    print(f"[SISTEMA] Injetando a inteligência artificial do arquivo: {caminho_pesos}...")
    
    checkpoint = torch.load(caminho_pesos, map_location=device, weights_only=False)
    
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

    # Carrega os pesos ignorando possíveis erros de keys missing se a arquitetura divergir um pouco
    modelo.load_state_dict(novo_state_dict, strict=False)
    modelo.to(device)
    modelo.eval()

    base_dir = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\dataset_processado\AffectNet"
    
    # Vamos avaliar o conjunto de validação ('val')
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
            print(f"\n[AVISO] A pasta {cenario} não existe! Verifique o caminho.")

if __name__ == "__main__":
    executar_pipeline()