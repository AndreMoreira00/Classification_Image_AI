# Classification_Image_AI

## Resumo do Projeto

Este projeto é uma implementação de um classificador de imagens utilizando técnicas de Aprendizado Profundo (Deep Learning). Ele emprega uma arquitetura de rede neural sequencial para classificar imagens de cachorros e gatos.

## Bibliotecas Utilizadas

- [tensorflow]
- [keras]
- [numpy]
- [matplotlib]

## Funções Utilizadas

### `load_data(directory)`
Esta função carrega e pré-processa os dados de imagens a partir de um diretório, convertendo as imagens em tensores e normalizando os valores dos pixels.

### `create_model(input_shape, num_classes)`
Esta função cria uma arquitetura de modelo de rede neural convolucional (CNN) usando a API Keras. A arquitetura específica pode variar dependendo dos parâmetros fornecidos.

### `train_model(model, X_train, y_train, X_val, y_val)`
Esta função treina o modelo de classificação de imagens usando os dados de treinamento e validação fornecidos. O treinamento é realizado usando o otimizador Adam e a função de perda de entropia cruzada categórica.

### `evaluate_model(model, X_test, y_test)`
Esta função avalia o desempenho do modelo de classificação de imagens usando os dados de teste fornecidos e retorna métricas de avaliação, como precisão e perda.

## Detalhes do Projeto

Este projeto visa classificar imagens em categorias específicas usando técnicas de Aprendizado Profundo. Ele explora arquiteturas de redes neurais convolucionais e técnicas de pré-processamento de imagem para alcançar seu objetivo.

## Exemplo de Uso

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from utils import load_data, create_model, train_model, evaluate_model

# Carregar e pré-processar os dados
X_train, y_train, X_val, y_val, X_test, y_test = load_data('data_directory')

# Criar o modelo
input_shape = X_train[0].shape
num_classes = len(np.unique(y_train))
model = create_model(input_shape, num_classes)

# Treinar o modelo
history = train_model(model, X_train, y_train, X_val, y_val)

# Avaliar o modelo
evaluate_model(model, X_test, y_test)
```
## Contribuindo
Contribuições são bem-vindas! Para contribuir com este projeto, por favor, abra uma issue ou envie um pull request.

## Licença
Este projeto é licenciado sob a MIT License.
