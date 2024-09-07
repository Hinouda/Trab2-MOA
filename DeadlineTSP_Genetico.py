import glob
import sys
import numpy as np
import time
import random
import greedyDL
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
                return float(1)

            if t[v] > self.prazos[v]:
                  return float(2)

            total_cost += self.matrizCustos[u][v]
            visited.add(v)

            t[v] = total_cost

        if len(visited) != self.numVertices or permutation[0] != 0 or permutation[-1] != 0:
            return float('3')

        return total_cost

def le_solucao_exata():
    arqv = open("solucaoExata.txt", "r")
    resolucao = []
    for linha in arqv:
        resolucao.append(linha.rstrip("\n"))

    return resolucao

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

    for instancia, idExata in zip(nomeInstancias, range(10)):
        numVertices, custos, arestas, prazos = leituraInstancia(instancia)

        print(f"\n\nResolucao do Caixeiro Viajante com Prazos para a instancia: {instancia}")
        arqv.write(f"\n\nResolucao do Caixeiro Viajante com Prazos para a instancia: {instancia}\n")

        decoder = TSPDecoder(numVertices, custos, prazos, arestas)

        brkga_params, _ = load_configuration("config.conf")

        brkga_params.population_size = min(brkga_params.population_size,
                                       10 * numVertices)

        brkga = BrkgaMpIpr(
            decoder=decoder,
            sense=Sense.MINIMIZE,
            seed=seed,  
            chromosome_size=numVertices,
            params=brkga_params
        )

        
        initial_cost, initial_tour = greedyDL.heuristica_insercao_mais_barata(numVertices, custos, prazos)

        keys = sorted([random.random() for _ in range(numVertices)])

        initial_chromosome = [0] * numVertices
        for i in range(numVertices):
            initial_chromosome[initial_tour[i]] = keys[i]

        brkga.set_initial_population([initial_chromosome])

        brkga.initialize()

        iter_without_improvement = 0
        best_cost = initial_cost * 2

        start_time = time.time()
        exec_time = 0
        stop_argument = 5000
        
        for iteration in range(num_generations):
            brkga.evolve(1)
            current_best_cost = brkga.get_best_fitness()

            if current_best_cost < best_cost:
                best_cost = current_best_cost
                exec_time = time.time() - start_time
                iter_without_improvement = 0  # Reset contador se houve melhora
            else:
                iter_without_improvement += 1

            print(f"Custo encontrado: {best_cost}")
            print(f"Iteracao {iteration}: {iter_without_improvement} iteracoes sem melhoria\n")
            

            # CONDIÇÕES DE PARADA: SEM MELHORIA OU COMPARAÇÃO SOLUÇÃO EXATA
            if stop_rule == StopRule.IMPROVEMENT or iter_without_improvement >= stop_argument:
                print(f"Criterio de {stop_argument} iteracoes sem melhoria atingido. Parando a execucao.")
                break
            
            resolucaoExata = le_solucao_exata()
            if resolucaoExata[idExata].count('.') == 0:
                if current_best_cost == int(resolucaoExata[idExata]):
                    print(f"Solucao encontrada eh a solução otima!")
                    break
                elif current_best_cost - int(resolucaoExata[idExata]) <=  int(resolucaoExata[idExata])  * 0.10:
                    print(f"Solucao encontrada esta 5% da solucao otima!")
                    break
        
        # MOSTRA MELHOR CUSTO E TEMPO DE EXECUÇÃO ATÉ ENCONTRAR O MELHOR CUSTO
        print(f"Melhor custo encontrado: {best_cost}")
        print(f"Tempo de execucao ate encontrar melhor custo: {exec_time}\n")
        arqv.write(f"Melhor custo encontrado: {best_cost}\n")
        arqv.write(f"Tempo de execucao ate encontrar melhor custo: {exec_time}\n")

        # Mostrar a melhor rota encontrada
        best_chromosome = brkga.get_best_chromosome()
        best_permutation = [0] + sorted(range(1, numVertices), key=lambda k: best_chromosome[k-1]) + [0]
        print(f"Melhor rota: {best_permutation}")
        arqv.write(f"Melhor rota: {best_permutation}\n")

    arqv.close()

if __name__ == "__main__":
    main()
