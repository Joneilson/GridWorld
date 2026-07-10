"""
grid_world.py — Ambiente Grid World de coleta de recursos.

Implementa a classe GridWorldEnv seguindo EXATAMENTE a seção 2 do SPEC.md.

Convenção de coordenadas
------------------------
Posição é uma tupla (x, y), como pede o SPEC (seção 2.2):
    - x = coluna (0 à esquerda, cresce para a direita)
    - y = linha  (0 no topo,     cresce para baixo)

Portanto os limites válidos são: 0 <= x < N e 0 <= y < N.

Ações (seção 2.3):
    0 = cima     -> y - 1
    1 = baixo    -> y + 1
    2 = esquerda -> x - 1
    3 = direita  -> x + 1

Elementos do grid (seção 2.1):
    A = agente     (1 por episódio)
    R = recurso    (padrão 5, fixos por seed)
    X = armadilha  (padrão 6, fixas por seed, terminam o episódio em falha)
    S = saída      (1, só é válida depois de coletar TODOS os recursos)
"""

import random


# Ações discretas (seção 2.3). O deslocamento é em (dx, dy).
ACOES = {
    0: (0, -1),   # cima
    1: (0, +1),   # baixo
    2: (-1, 0),   # esquerda
    3: (+1, 0),   # direita
}

NOME_ACAO = {0: "cima", 1: "baixo", 2: "esquerda", 3: "direita"}


class GridWorldEnv:
    """
    Ambiente de coleta de recursos em grid N x N com energia limitada.

    O layout (posição inicial do agente, saída, recursos e armadilhas) é
    sorteado UMA vez a partir da `seed` e fica FIXO entre chamadas de reset()
    — isso implementa a recomendação da seção 6 (item 3): ambiente fixo por
    seed durante treino/comparação, para garantir comparação justa entre os
    três agentes.

    Cada reset() apenas recoloca o agente na posição inicial, restaura a
    energia, zera os recursos coletados e o contador de passos.
    """

    def __init__(
        self,
        N=10,
        n_recursos=5,
        n_armadilhas=6,
        energia_inicial=None,
        max_passos=200,
        seed=None,
    ):
        self.N = N
        self.n_recursos = n_recursos
        self.n_armadilhas = n_armadilhas
        # Padrão da seção 2.6: 3 * N * N dá folga para explorar sem ser trivial.
        self.energia_inicial = energia_inicial if energia_inicial is not None else 3 * N * N
        self.max_passos = max_passos
        self.seed = seed

        # Gera o layout fixo do cenário a partir da seed.
        self._gerar_layout(seed)

        # Estado dinâmico (definido de fato em reset()).
        self.pos = None
        self.coletados = 0          # bitmask: bit i = recurso i já coletado
        self.energia = 0
        self.passos = 0
        self.terminado = False

    # ------------------------------------------------------------------ #
    # Construção do cenário
    # ------------------------------------------------------------------ #
    def _gerar_layout(self, seed):
        """Sorteia posições distintas para agente, saída, recursos e armadilhas."""
        rng = random.Random(seed)

        n_celulas_necessarias = 2 + self.n_recursos + self.n_armadilhas
        if n_celulas_necessarias > self.N * self.N:
            raise ValueError(
                f"Cenário impossível: preciso de {n_celulas_necessarias} células "
                f"distintas, mas o grid {self.N}x{self.N} só tem {self.N * self.N}."
            )

        # Todas as células do grid, embaralhadas; consumimos sem repetição.
        todas = [(x, y) for y in range(self.N) for x in range(self.N)]
        rng.shuffle(todas)

        self.pos_inicial = todas.pop()
        self.pos_saida = todas.pop()
        # A ordem da lista define o bit de cada recurso no bitmask.
        self.recursos = [todas.pop() for _ in range(self.n_recursos)]
        self.armadilhas = set(todas.pop() for _ in range(self.n_armadilhas))

        # Máscara com todos os recursos coletados (ex.: 5 recursos -> 0b11111).
        self.mascara_completa = (1 << self.n_recursos) - 1

    # ------------------------------------------------------------------ #
    # API principal: reset / step / render
    # ------------------------------------------------------------------ #
    def reset(self):
        """Reinicia o episódio e devolve o estado inicial."""
        self.pos = self.pos_inicial
        self.energia = self.energia_inicial
        self.coletados = 0
        self.passos = 0
        self.terminado = False
        return self._get_estado()

    def step(self, acao):
        """
        Executa uma ação e devolve (estado, recompensa, terminado, info).

        Segue a tabela de recompensas e as condições de término da seção 2.
        """
        if self.terminado:
            # Proteção: episódio já acabou; não altera mais nada.
            return self._get_estado(), 0.0, True, {"aviso": "episodio ja terminou"}
        if acao not in ACOES:
            raise ValueError(f"Ação inválida: {acao}. Use 0,1,2,3.")

        # 1) Move (ou permanece, se bateu na borda). Consumo de energia acontece
        #    SEMPRE, inclusive na tentativa de movimento inválido (seção 2.3).
        dx, dy = ACOES[acao]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        if 0 <= nx < self.N and 0 <= ny < self.N:
            self.pos = (nx, ny)          # movimento válido
        # senão: permanece no lugar (movimento inválido), mas ainda gasta energia.

        self.passos += 1
        self.energia -= 1

        # 2) Avalia o evento resultante do movimento (recompensa base + término).
        recompensa, info = self._avaliar_celula()

        # 3) Se o passo não terminou por evento do grid, checa exaustão e timeout.
        #    A penalidade de -20 SOMA-SE à recompensa do passo (não sobrescreve):
        #    assim uma coleta feita no último passo de energia ainda é creditada
        #    (+10 - 20 = -10). Isso preserva o sinal de aprendizado para RL/GA —
        #    "chegar naquela célula" continua valendo a pena.
        if not self.terminado:
            if self.energia <= 0:
                recompensa += -20.0
                self.terminado = True
                info = {**info, "motivo_termino": "exaustao", "sucesso": False}
            elif self.passos >= self.max_passos:
                recompensa += -20.0
                self.terminado = True
                info = {**info, "motivo_termino": "timeout", "sucesso": False}

        return self._get_estado(), recompensa, self.terminado, info

    def _avaliar_celula(self):
        """
        Aplica as regras da célula em que o agente acabou de entrar.

        Retorna (recompensa, info) e ajusta self.coletados / self.terminado.
        """
        # Armadilha: fim de episódio em falha (seção 2.4 e 2.5).
        if self.pos in self.armadilhas:
            self.terminado = True
            return -50.0, {"motivo": "armadilha", "sucesso": False}

        # Recurso ainda não coletado: coleta e ganha +10.
        for i, celula in enumerate(self.recursos):
            if self.pos == celula and not (self.coletados & (1 << i)):
                self.coletados |= (1 << i)
                return +10.0, {"motivo": "coleta", "recurso": i, "sucesso": False}

        # Saída: só é sucesso se TODOS os recursos já foram coletados.
        if self.pos == self.pos_saida:
            if self.coletados == self.mascara_completa:
                self.terminado = True
                return +100.0, {"motivo": "saida_sucesso", "sucesso": True}
            # Saída sem todos os recursos: trata como passo comum (agente não sai).
            return -1.0, {"motivo": "saida_incompleta", "sucesso": False}

        # Passo comum: custo de energia.
        return -1.0, {"motivo": "passo", "sucesso": False}

    def render(self, mode="human"):
        """
        Desenha o grid em ASCII. Recursos já coletados somem do mapa.

        Prioridade de exibição numa mesma célula: A (agente) > demais símbolos.
        Retorna a string do mapa; se mode == "human", também imprime.
        """
        # Índice rápido dos recursos ainda não coletados.
        recursos_pendentes = {
            celula for i, celula in enumerate(self.recursos)
            if not (self.coletados & (1 << i))
        }

        linhas = []
        for y in range(self.N):
            linha = []
            for x in range(self.N):
                celula = (x, y)
                if celula == self.pos:
                    ch = "A"
                elif celula in self.armadilhas:
                    ch = "X"
                elif celula == self.pos_saida:
                    ch = "S"
                elif celula in recursos_pendentes:
                    ch = "R"
                else:
                    ch = "."
                linha.append(ch)
            linhas.append(" ".join(linha))

        cabecalho = (
            f"passos={self.passos}  energia={self.energia}  "
            f"coletados={self.coletados:0{self.n_recursos}b} "
            f"({bin(self.coletados).count('1')}/{self.n_recursos})  "
            f"terminado={self.terminado}"
        )
        mapa = cabecalho + "\n" + "\n".join(linhas)

        if mode == "human":
            print(mapa)
        return mapa

    # ------------------------------------------------------------------ #
    # Auxiliares
    # ------------------------------------------------------------------ #
    def _get_estado(self):
        """
        Estado/observação da seção 2.2:
            (posicao_agente (x, y), recursos_coletados bitmask, energia_restante)
        É uma tupla hashable — serve direto como chave de Q-table.
        """
        return (self.pos, self.coletados, self.energia)

    @property
    def n_acoes(self):
        return len(ACOES)


# ---------------------------------------------------------------------- #
# Autoteste rápido: roda um agente aleatório por alguns passos e mostra o
# ambiente funcionando ponta a ponta. Execute:  python -m ambiente.grid_world
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    env = GridWorldEnv(seed=42)
    estado = env.reset()

    print("=== Estado inicial ===")
    print("estado:", estado)
    env.render()

    print("\n=== Passeando aleatoriamente (10 passos) ===")
    rng = random.Random(0)
    for t in range(10):
        acao = rng.randrange(env.n_acoes)
        estado, recompensa, terminado, info = env.step(acao)
        print(
            f"\npasso {t + 1}: acao={acao}({NOME_ACAO[acao]})  "
            f"recompensa={recompensa}  info={info}"
        )
        env.render()
        if terminado:
            print("\n>>> episódio terminado:", info)
            break
