import numpy as np

def calcular_custo_insercao(rota, cidade, matrizCustos):
    menor_custo = float('inf')
    melhor_posicao = -1

    # Tentar inserir a cidade em todas as posições possíveis
    for i in range(1, len(rota)):
        custo_insercao = (
            matrizCustos[rota[i-1]][cidade] + 
            matrizCustos[cidade][rota[i]] - 
            matrizCustos[rota[i-1]][rota[i]]
        )
        if custo_insercao < menor_custo:
            menor_custo = custo_insercao
            melhor_posicao = i
    
    return melhor_posicao, menor_custo

def heuristica_insercao_mais_barata(numVertices, matrizCustos, prazos):
    
    # Inicializar a rota com dois vértices: 0 e o vértice com menor custo
    rota = [0, np.argmin(matrizCustos[0][1:]) + 1, 0]
    visitados = {0, rota[1]}  # Conjunto de cidades já visitadas
    custo_total = matrizCustos[0][rota[1]] + matrizCustos[rota[1]][0]
    tempo_atual = custo_total

    while len(visitados) < numVertices:
        melhor_insercao = None
        melhor_custo = float('inf')

        # Verificar qual vértice não visitado pode ser inserido
        for cidade in range(numVertices):
            if cidade not in visitados:
                posicao_insercao, custo_insercao = calcular_custo_insercao(rota, cidade, matrizCustos)
                
                # Verificar se a inserção respeita o prazo
                tempo_chegada = tempo_atual + custo_insercao
                if tempo_chegada <= prazos[cidade] and custo_insercao < melhor_custo:
                    melhor_insercao = (cidade, posicao_insercao)
                    melhor_custo = custo_insercao

        # Se não encontrou inserção válida, relaxe a condição de prazos
        if melhor_insercao is None:
            for cidade in range(numVertices):
                if cidade not in visitados:
                    posicao_insercao, custo_insercao = calcular_custo_insercao(rota, cidade, matrizCustos)
                    melhor_insercao = (cidade, posicao_insercao)
                    melhor_custo = custo_insercao
                    break

        # Inserir a cidade na posição ótima
        cidade, posicao = melhor_insercao
        rota.insert(posicao, cidade)
        visitados.add(cidade)
        
        # Atualizar o custo e o tempo atual
        custo_total += melhor_custo
        tempo_atual += melhor_custo

    return custo_total, rota