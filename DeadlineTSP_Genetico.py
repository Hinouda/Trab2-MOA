import glob
import sys
import numpy as np
import time
from brkga_mp_ipr.enums import Sense
from brkga_mp_ipr.algorithm import BrkgaMpIpr
from brkga_mp_ipr.types_io import load_configuration
from brkga_mp_ipr.types import BaseChromosome

# Função para leitura de instância
def leituraInstancia(nomeInstancia):
    arqv = open(nomeInstancia, "r")

    numVertices = int(arqv.readline())
    matrizCustos = np.empty((numVertices, numVertices), dtype=int)
    
    for i in range(numVertices):
        linha = list(map(int, arqv.readline().split()))
        for j in range(numVertices):
            matrizCustos[i][j] = linha[j]
    arqv.readline()  # Linha vazia entre a matriz e os prazos
    prazos = list(map(int, arqv.readline().split()))

    arestas = [(i, j) for i in range(numVertices) for j in range(numVertices) if matrizCustos[i][j] != 0]

    return numVertices, matrizCustos, arestas, prazos

# Classe de decoder para o problema do Caixeiro Viajante
class TSPDecoder:
    def __init__(self, numVertices, matrizCustos, prazos, arestas):
        self.numVertices = numVertices
        self.matrizCustos = matrizCustos
        self.prazos = prazos
        self.arestas = set(arestas)  
        self.M = 100000  # Valor grande para penalizar subrotas

    def decode(self, chromosome: BaseChromosome, rewrite: bool = False) -> float:
        permutation = [0] + sorted(range(1, self.numVertices), key=lambda k: chromosome[k-1]) + [0]

        total_cost = 0
        time = 0
        t = [0] * self.numVertices
        visited = set()
        visited.add(0)

        for k in range(len(permutation) - 1):
            u = permutation[k]
            v = permutation[k + 1]

            if (u, v) not in self.arestas:
                return float('inf')

            time += self.matrizCustos[u][v]
            t[v] = time

            if v != 0 and t[v] > self.prazos[v]:
                return float('inf')

            total_cost += self.matrizCustos[u][v]
            visited.add(v)

        if len(visited) != self.numVertices or permutation[0] != 0 or permutation[-1] != 0:
            return float('inf')

        return total_cost

# Função para validar o custo da permutação gerada
def calcular_custo_rota(permutacao, matrizCustos):
    total_cost = 0
    for i in range(len(permutacao) - 1):
        u = permutacao[i]
        v = permutacao[i + 1]
        total_cost += matrizCustos[u][v]
    return total_cost

# Critérios de parada
class StopRule:
    GENERATIONS = 0
    TARGET = 1
    IMPROVEMENT = 2

# Função principal
def main():
    arqv = open("testes.txt", "w")
    if len(sys.argv) < 3:
        print("Usage: python main_minimal.py <seed> <num_generations> ")
        sys.exit(1)

    seed = int(sys.argv[1])
    num_generations  = int(sys.argv[2])
    stop_rule = StopRule.GENERATIONS  

    nomeInstancias = glob.glob('Instancias/*.txt')

    for instancia in nomeInstancias:
        numVertices, custos, arestas, prazos = leituraInstancia(instancia)

        print(f"\n\nResolução do Caixeiro Viajante com Prazos para a instância: {instancia}")
        arqv.write(f"\n\nResolução do Caixeiro Viajante com Prazos para a instância: {instancia}\n")

        decoder = TSPDecoder(numVertices, custos, prazos, arestas)

        brkga_params, _ = load_configuration("config.conf")

        brkga = BrkgaMpIpr(
            decoder=decoder,
            sense=Sense.MINIMIZE,
            seed=seed,  
            chromosome_size=numVertices,
            params=brkga_params
        )

        brkga.initialize()

        iter_without_improvement = 0
        best_cost = float('inf')
        target_cost = 0 

        start_time = time.time()
        stop_argument = 200 if numVertices <= 20 else 500
        maximum_time = 12000  

        for iteration in range(num_generations):
            brkga.evolve(1)
            current_best_cost = brkga.get_best_fitness()

            if current_best_cost < best_cost:
                best_cost = current_best_cost
                iter_without_improvement = 0  # Reset contador se houve melhora
            else:
                iter_without_improvement += 1

            print(current_best_cost)
            print(f"Iteração {iteration}: {iter_without_improvement} iterações sem melhoria")

            # Verificar critérios de parada
            if time.time() - start_time > maximum_time:
                print("Critério de tempo atingido. Parando a execução.")
                break

            if stop_rule == StopRule.IMPROVEMENT or iter_without_improvement >= stop_argument:
                print(f"Critério de {stop_argument} iterações sem melhoria atingido. Parando a execução.")
                break

            if stop_rule == StopRule.TARGET or best_cost <= target_cost:
                print(f"Critério de atingir o custo alvo {target_cost} atingido. Parando a execução.")
                break

        print(f"Melhor custo encontrado: {best_cost}")
        arqv.write(f"Melhor custo encontrado: {best_cost}\n")

        # Mostrar a melhor rota encontrada
        best_chromosome = brkga.get_best_chromosome()
        best_permutation = [0] + sorted(range(1, numVertices), key=lambda k: best_chromosome[k-1]) + [0]
        print(f"Melhor rota: {best_permutation}")
        arqv.write(f"Melhor rota: {best_permutation}\n")

        # Validação final da rota
        custo_calculado = calcular_custo_rota(best_permutation, custos)
        if custo_calculado == best_cost:
            print(f"A rota {best_permutation} corresponde ao custo {best_cost}.")
            arqv.write(f"A rota {best_permutation} corresponde ao custo {best_cost}.\n")
        else:
            print(f"Inconsistência encontrada! A rota {best_permutation} tem custo {custo_calculado}, mas o melhor custo encontrado foi {best_cost}.")
            arqv.write(f"Inconsistência encontrada! A rota {best_permutation} tem custo {custo_calculado}, mas o melhor custo encontrado foi {best_cost}.\n")

    arqv.close()

if __name__ == "__main__":
    main()
