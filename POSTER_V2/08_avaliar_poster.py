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
from tqdm import tqdm # <--- ADICIONAMOS A BARRA DE PROGRESSO AQUI

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
        dataset = datasets.ImageFolder(root=caminho_pasta, transform=transformacoes)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0) 
    except Exception as e:
        print(f"[ERRO] Não foi possível carregar as imagens em {caminho_pasta}: {e}")
        return

    classes = dataset.classes
    todas_preds = []
    todas_labels = []

    print(f"[INFO] Lendo imagens de {caminho_pasta}...")
    
    with torch.no_grad():
        # <--- ADICIONAMOS O TQDM AQUI PARA VOCÊ VER A MÁGICA ACONTECER --->
        for imagens, labels in tqdm(dataloader, desc=f"Inferência ({nome_cenario})", unit="lote"):
            imagens = imagens.to(device)
            labels = labels.to(device)

            saida = modelo(imagens)
            
            if isinstance(saida, tuple):
                saida = saida[0]
                
            _, preds = torch.max(saida, 1)
            
            todas_preds.extend(preds.cpu().numpy())
            todas_labels.extend(labels.cpu().numpy())

    if len(todas_preds) == 0:
        print("[AVISO] Nenhuma predição feita. A pasta de teste está vazia?")
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
    device = torch.device("cpu")
    print(f"\n[SISTEMA] Iniciando processamento via: {str(device).upper()} (Modo de Compatibilidade)")

    print("[SISTEMA] Montando a arquitetura POSTER_V2 (pyramid_trans_expr2)...")
    modelo = pyramid_trans_expr2(img_size=224, num_classes=7)
    
    caminho_pesos = r"./checkpoint/raf-db-model_best.pth" 
    
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

    modelo.load_state_dict(novo_state_dict)
    modelo.to(device)
    modelo.eval()

    base_dir = r"C:\Users\Gustavo\Documents\ravdess-emotion-demo\dataset_processado\RAF-DB"
    cenarios = [
        os.path.join(base_dir, "Baseline", "test"),
        os.path.join(base_dir, "HMD", "test"),
        os.path.join(base_dir, "HMD_Olhos", "test"),
        os.path.join(base_dir, "HMD_Olhos_Reais", "test")
    ]

    for cenario in cenarios:
        if os.path.exists(cenario):
            avaliar_cenario(modelo, cenario, device)
        else:
            print(f"\n[AVISO] A pasta {cenario} não existe! Verifique o caminho.")

if __name__ == "__main__":
    executar_pipeline()