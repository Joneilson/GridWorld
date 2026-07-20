"""
agente_genetico.py — Agente evoluído por Algoritmo Genético (GA). Seção 3.3.

Diferente do Q-Learning (que aprende por atualização de valor), o GA NÃO usa
recompensa passo a passo para ajustar nada: mantém uma POPULAÇÃO de indivíduos e
os evolui por seleção, cruzamento e mutação, guiado apenas pelo FITNESS
(recompensa total de rodar o indivíduo no ambiente).

Representação do indivíduo: SEQUÊNCIA DE AÇÕES (a alternativa da seção 3.3)
--------------------------------------------------------------------------
Cada indivíduo é uma lista fixa de `comprimento` ações — gene t = ação a tomar
no passo t (política "de malha aberta": ignora o estado, segue o plano). O SPEC
sugere duas representações; a de sequência foi ESCOLHIDA por uma razão empírica:

    A representação alternativa — tabela ação-por-estado (pos, bitmask) — só
    coloca sob seleção os genes dos estados VISITADOS; estender a trajetória cai
    sempre em genes nunca otimizados, o que trava o GA num ótimo local ("coleta 1
    recurso e morre na armadilha"). Nos testes, a tabela estagnava e NÃO resolvia
    nenhum mapa. A sequência otimiza a trajetória inteira de forma coerente e
    resolve o mapa padrão (seed=42), além de tornar a evolução fácil de VISUALIZAR
    (exigido no item 4/8 do vídeo).

Trade-off (documentar no item 9 do vídeo): a sequência é um PLANO FIXO, então
superajusta a um mapa específico — trocar de mapa exige re-evoluir. É válida no
protocolo de mapa fixo (seção 6), mas menos "comparável" ao Q-Learning, que
aprende uma política dependente de estado.

Formulação computacional (item 6 do vídeo):
    - Indivíduo (genoma) : lista de `comprimento` ações 0..3 (gene = ação no tempo t)
    - Aptidão (fitness)  : recompensa total ao rodar a sequência no ambiente (seção 2.4)
    - Seleção            : torneio (tamanho 3 por padrão)
    - Cruzamento         : uniforme (cada gene vem de um dos dois pais, 50/50)
    - Mutação            : por gene, com probabilidade p_mut, sorteia nova ação
    - Elitismo           : os N melhores passam intactos para a geração seguinte
"""

import random

from ambiente.grid_world import GridWorldEnv
from agentes.agente_aleatorio import rodar_episodio


class AgenteGenetico:
    """
    Evolui uma sequência de ações por GA. Segue a interface de agente:
        reset()                 -> reinicia o ponteiro do plano (novo episódio)
        escolher_acao(estado)   -> próxima ação do MELHOR plano (ignora o estado)

    Treino: chame treinar(). O melhor plano fica em self.melhor.

    Hiperparâmetros (SPEC 3.3 dá valores iniciais; ajustados empiricamente para
    resolver o mapa padrão):
        comprimento     80     nº de ações do plano (horizonte)
        tam_populacao   200    indivíduos por geração
        n_geracoes      500    gerações de evolução
        taxa_mutacao    0.05   probabilidade de mutar CADA gene
        taxa_crossover  0.70   probabilidade de cruzar (senão o filho clona um pai)
        tam_torneio     3      candidatos disputados por vaga na seleção
        elitismo        2      melhores preservados intactos a cada geração
    """

    def __init__(self, env, comprimento=80, tam_populacao=200, n_geracoes=500,
                 taxa_mutacao=0.05, taxa_crossover=0.7, tam_torneio=3,
                 elitismo=2, seed=None):
        self.env = env
        self.n_acoes = env.n_acoes
        self.comprimento = comprimento

        self.tam_populacao = tam_populacao
        self.n_geracoes = n_geracoes
        self.taxa_mutacao = taxa_mutacao
        self.taxa_crossover = taxa_crossover
        self.tam_torneio = tam_torneio
        self.elitismo = elitismo
        self.rng = random.Random(seed)

        # Melhor indivíduo global (a "solução" do GA) e curvas de evolução.
        self.melhor = None
        self.melhor_fitness = float("-inf")
        self.historico_fitness_max = []
        self.historico_fitness_medio = []
        self.treinado = False

        self._idx = 0                  # ponteiro do plano durante a inferência

    # ------------------------------------------------------------------ #
    # Interface de agente (inferência)
    # ------------------------------------------------------------------ #
    def reset(self):
        self._idx = 0

    def escolher_acao(self, estado):
        # Política de malha aberta: segue o plano, ignorando o estado.
        if self.melhor is None:
            raise RuntimeError("Agente genético ainda não treinado — chame treinar().")
        if self._idx < len(self.melhor):
            acao = self.melhor[self._idx]
        else:
            acao = 0                   # plano esgotado (raro num plano bom): segue reto
        self._idx += 1
        return acao

    # ------------------------------------------------------------------ #
    # Blocos do GA
    # ------------------------------------------------------------------ #
    def _genoma_aleatorio(self):
        """Um plano inicial: `comprimento` ações aleatórias."""
        return [self.rng.randrange(self.n_acoes) for _ in range(self.comprimento)]

    def _fitness(self, genoma):
        """Recompensa total ao rodar a sequência `genoma` no ambiente."""
        self.env.reset()
        total = 0.0
        for acao in genoma:
            _estado, recompensa, terminado, _info = self.env.step(acao)
            total += recompensa
            if terminado:
                break
        return total

    def _selecionar(self, populacao, fitnesses):
        """Seleção por torneio: sorteia `tam_torneio` e devolve o de maior fitness."""
        melhor_i = self.rng.randrange(len(populacao))
        for _ in range(self.tam_torneio - 1):
            i = self.rng.randrange(len(populacao))
            if fitnesses[i] > fitnesses[melhor_i]:
                melhor_i = i
        return populacao[melhor_i]

    def _crossover(self, pai1, pai2):
        """Cruzamento uniforme: cada gene vem de um dos pais (50/50)."""
        if self.rng.random() > self.taxa_crossover:
            return list(pai1)          # sem cruzamento: filho clona o pai1
        escolha = self.rng.random
        return [g1 if escolha() < 0.5 else g2 for g1, g2 in zip(pai1, pai2)]

    def _mutar(self, genoma):
        """Mutação por gene: com prob. taxa_mutacao, sorteia nova ação."""
        for i in range(len(genoma)):
            if self.rng.random() < self.taxa_mutacao:
                genoma[i] = self.rng.randrange(self.n_acoes)
        return genoma

    # ------------------------------------------------------------------ #
    # Loop evolutivo
    # ------------------------------------------------------------------ #
    def treinar(self, log_cada=0):
        """
        Evolui a população por n_geracoes e devolve (fitness_max, fitness_medio)
        por geração — as curvas de evolução (gráfico exigido no item 5/8 do vídeo).
        O melhor indivíduo encontrado fica em self.melhor.
        """
        populacao = [self._genoma_aleatorio() for _ in range(self.tam_populacao)]
        self.historico_fitness_max = []
        self.historico_fitness_medio = []

        for ger in range(self.n_geracoes):
            fitnesses = [self._fitness(ind) for ind in populacao]

            f_max = max(fitnesses)
            f_med = sum(fitnesses) / len(fitnesses)
            self.historico_fitness_max.append(f_max)
            self.historico_fitness_medio.append(f_med)

            # Guarda o melhor global (a solução devolvida ao final).
            i_melhor = max(range(len(populacao)), key=lambda i: fitnesses[i])
            if f_max > self.melhor_fitness:
                self.melhor = list(populacao[i_melhor])
                self.melhor_fitness = f_max

            if log_cada and (ger + 1) % log_cada == 0:
                print(f"  geração {ger + 1:>4}/{self.n_geracoes}  "
                      f"fitness máx: {f_max:+.1f}   médio: {f_med:+.1f}")

            # Nova geração: elitismo + descendentes (seleção → crossover → mutação).
            ordem = sorted(range(len(populacao)),
                           key=lambda i: fitnesses[i], reverse=True)
            nova = [list(populacao[i]) for i in ordem[:self.elitismo]]
            while len(nova) < self.tam_populacao:
                pai1 = self._selecionar(populacao, fitnesses)
                pai2 = self._selecionar(populacao, fitnesses)
                filho = self._crossover(pai1, pai2)
                self._mutar(filho)
                nova.append(filho)
            populacao = nova

        self.treinado = True
        return self.historico_fitness_max, self.historico_fitness_medio


# ---------------------------------------------------------------------- #
# Curva de evolução (gráfico exigido no item 5/8 do vídeo)
# ---------------------------------------------------------------------- #
def salvar_curva_evolucao(fitness_max, fitness_medio,
                          caminho="resultados/graficos/curva_genetico.png",
                          titulo="Evolução do fitness — Algoritmo Genético"):
    """
    Plota fitness máximo e médio por geração e salva a figura em `caminho`.
    matplotlib é importado de forma preguiçosa (o resto do módulo é stdlib puro).
    Devolve o caminho salvo, ou None se matplotlib não estiver instalado.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [aviso] matplotlib não instalado — curva pulada. "
              "Instale com: pip install matplotlib")
        return None

    import os
    pasta = os.path.dirname(caminho)
    if pasta:
        os.makedirs(pasta, exist_ok=True)

    geracoes = range(1, len(fitness_max) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(geracoes, fitness_max, linewidth=2.0, color="crimson",
             label="fitness máximo")
    plt.plot(geracoes, fitness_medio, linewidth=1.5, color="steelblue",
             label="fitness médio")
    plt.xlabel("geração")
    plt.ylabel("fitness (recompensa total)")
    plt.title(titulo)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=120)
    plt.close()
    return caminho


# ---------------------------------------------------------------------- #
# Autoteste: evolui no mapa seed=42, salva a curva e compara com o baseline.
# Execute:  python -m agentes.agente_genetico
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    from agentes.agente_aleatorio import AgenteAleatorio

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    SEED_MAPA = 42
    env = GridWorldEnv(seed=SEED_MAPA)
    agente = AgenteGenetico(env, seed=0)

    print(f"=== Evoluindo GA: pop={agente.tam_populacao}, "
          f"gerações={agente.n_geracoes}, plano={agente.comprimento} "
          f"(mapa seed={SEED_MAPA}) ===")
    agente.treinar(log_cada=max(1, agente.n_geracoes // 10))
    print(f"\nmelhor fitness encontrado: {agente.melhor_fitness:+.1f}")

    caminho = salvar_curva_evolucao(agente.historico_fitness_max,
                                    agente.historico_fitness_medio)
    if caminho:
        print(f"curva de evolução salva em: {caminho}")

    m = rodar_episodio(env, agente)
    base = rodar_episodio(env, AgenteAleatorio(n_acoes=env.n_acoes, seed=0))
    print("\n=== Melhor indivíduo × baseline aleatório ===")
    print(f"  GA (evoluído) : sucesso={m['sucesso']}  recompensa={m['recompensa']:+.1f}  "
          f"passos={m['passos']}")
    print(f"  Aleatório     : sucesso={base['sucesso']}  recompensa={base['recompensa']:+.1f}  "
          f"passos={base['passos']}")
