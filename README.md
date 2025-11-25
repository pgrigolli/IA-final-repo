# Projeto Classificador do Simpsons

Alunos:
Artur Bento de Carvalho
Pedro Chouery Grigolli
Thiago Riemma Carbonera

Este projeto realiza classificação de personagens dos Simpsons usando Deep Features (Xception + ViT) e 20 classificadores tradicionais do scikit-learn, além de um modelo Ensemble.

O pipeline completo:
1. Extrai deep features de cada imagem:
    - Xception (TensorFlow)
    - ViT-B/16 (timm – PyTorch)
2. Concatena as features em um vetor grande → feature vector final
3. Treina 20 classificadores (SVM, KNN, MLP, RF)
4. Avalia cada um (Accuracy + F1-score)
5. Treina um Ensemble Soft Voting
6. Gera a matriz de confusão final

- Resultados alcançados:
    - Ensemble Accuracy: ~82%
    - Ensemble F1-score: ~81%

## Como Executar

### Criar o Ambiente Virtual
No terminal:
```bash
python3 -m venv meuenv
source meuenv/bin/activate
```
### Instalar Dependências
Com o ambiente ativado:
```bash
pip install -r requirements.txt
```
- Pacotes instalados:
    - tensorflow
    - torch
    - timm
    - opencv-python
    - matplotlib
    - numpy
    - scikit-learn
    - tqdm
    - pillow
    - joblib

### Executar o Projeto
No terminal, com o ambiente ativado:
```bash
python main.py
```

- O programa irá:
    - Carregar Xception
    - Carregar ViT
    - Extrair features
    - Treinar 20 classificadores
    - Treinar Ensemble
    - Gerar matriz de confusão

Saída final salva em:
```
matriz_confusao_ensemble.png
```