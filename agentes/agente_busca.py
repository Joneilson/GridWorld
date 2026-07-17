"""
agente_busca.py — Agente de busca heurística A*. Seção 3.1 do SPEC.

Planejamento (NÃO aprendizado): a cada episódio, roda A* do zero sobre o espaço
de estados composto (posição + bitmask de recursos coletados) e produz uma
sequência de ações do início até o sucesso. Depois só "toca" esse plano.

Formulação computacional (bom material para o vídeo, item 6):
    - Estado de busca : (posicao, coletados_bitmask)   -- energia NÃO entra aqui
    - Ação            : 0..3 (cima/baixo/esquerda/direita)
    - Custo por ação  : 1  (bate com o -1 de "passo comum" do env, SPEC 3.1)
    - Objetivo (goal) : coletados == todos  E  posicao == saida
    - Armadilhas      : tratadas como PAREDES (o plano nunca entra nelas)

Por que energia não entra na busca? (SPEC 3.1) O espaço de estados é
posição × bitmask = N*N * 2^n_recursos. Com energia inicial = 3*N*N (=300 no
10x10), a energia é folgada demais para restringir o caminho ótimo — então é
seguro planejar sem ela e apenas assumir que há energia suficiente.
"""

import heapq

from ambiente.grid_world import GridWorldEnv, ACOES
from agentes.agente_aleatorio import rodar_episodio


def _manhattan(a, b):
    """Distância de Manhattan entre duas células (x, y)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class AgenteBusca:
    """
    Planejador A* sobre (posicao, bitmask). Segue a interface de agente:
    reset() (re)planeja; escolher_acao(estado) devolve a próxima ação do plano.

    Parâmetros (SPEC 3.1 pede expor a heurística):
        heuristica: "admissivel" (padrão) | "ingenua" | "guloso" | "nula"
    """

    HEURISTICAS = ("admissivel", "ingenua", "guloso", "nula")

    def __init__(self, env, heuristica="admissivel"):
        if heuristica not in self.HEURISTICAS:
            raise ValueError(
                f"Heurística '{heuristica}' inválida. Use uma de {self.HEURISTICAS}."
            )
        self.env = env
        self.heuristica = heuristica
        # Preenchidos por planejar() / reset():
        self.plano = None            # lista de ações do início até o sucesso
        self.idx = 0
        self.nos_expandidos = 0

    # ------------------------------------------------------------------ #
    # Interface de agente
    # ------------------------------------------------------------------ #
    def reset(self):
        """Replaneja do zero (é planejamento, não aprendizado — SPEC 3.1)."""
        self.plano = self.planejar()
        self.idx = 0

    def escolher_acao(self, estado):
        if self.plano is None:
            self.reset()
        if self.idx >= len(self.plano):
            # Plano esgotado (não deveria ocorrer num plano correto): fica parado.
            return 0
        acao = self.plano[self.idx]
        self.idx += 1
        return acao

    # ------------------------------------------------------------------ #
    # A*
    # ------------------------------------------------------------------ #
    def planejar(self):
        """
        Roda A* e devolve a lista de ações (ou None se não há solução).
        Também registra self.nos_expandidos (para comparar heurísticas).
        """
        env = self.env
        N = env.N
        recursos = env.recursos
        armadilhas = env.armadilhas
        saida = env.pos_saida
        meta_mask = env.mascara_completa

        inicio = (env.pos_inicial, 0)

        def eh_objetivo(pos, mask):
            return mask == meta_mask and pos == saida

        # g = melhor custo conhecido até o estado; came = de onde/como chegamos.
        g = {inicio: 0}
        came = {}                      # estado -> (estado_anterior, acao)
        contador = 0                   # desempate estável na fila de prioridade
        fila = [(self._h(*inicio), 0, contador, inicio)]
        visitados = set()
        self.nos_expandidos = 0

        while fila:
            _, g_atual, _, estado = heapq.heappop(fila)
            if estado in visitados:
                continue
            visitados.add(estado)
            pos, mask = estado

            if eh_objetivo(pos, mask):
                return self._reconstruir(came, estado)

            self.nos_expandidos += 1

            for acao, (dx, dy) in ACOES.items():
                nx, ny = pos[0] + dx, pos[1] + dy
                # Fora do grid: movimento inválido -> não gera sucessor útil.
                if not (0 <= nx < N and 0 <= ny < N):
                    continue
                # Armadilha = parede: o plano jamais entra numa.
                if (nx, ny) in armadilhas:
                    continue

                nova_pos = (nx, ny)
                nova_mask = mask
                # Se a célula é um recurso ainda não coletado, liga o bit.
                for i, r in enumerate(recursos):
                    if nova_pos == r:
                        nova_mask |= (1 << i)
                        break

                prox = (nova_pos, nova_mask)
                novo_g = g_atual + 1                       # custo por passo = 1
                if novo_g < g.get(prox, float("inf")):
                    g[prox] = novo_g
                    came[prox] = (estado, acao)
                    contador += 1
                    f = novo_g + self._h(nova_pos, nova_mask)
                    heapq.heappush(fila, (f, novo_g, contador, prox))

        return None                    # objetivo inatingível (ex.: saída murada)

    def _reconstruir(self, came, estado):
        """Refaz a sequência de ações do início até `estado`."""
        acoes = []
        while estado in came:
            anterior, acao = came[estado]
            acoes.append(acao)
            estado = anterior
        acoes.reverse()
        return acoes

    # ------------------------------------------------------------------ #
    # Heurísticas (todas devem devolver um LOWER BOUND para serem admissíveis)
    # ------------------------------------------------------------------ #
    def _h(self, pos, mask):
        env = self.env
        saida = env.pos_saida
        restantes = [r for i, r in enumerate(env.recursos) if not (mask & (1 << i))]

        if self.heuristica == "nula":
            return 0

        # Sem recursos pendentes: só falta chegar à saída (exato p/ Manhattan).
        if not restantes:
            return _manhattan(pos, saida)

        if self.heuristica == "ingenua":
            # Só distância até a saída. Ignora recursos -> admissível mas fraca.
            return _manhattan(pos, saida)

        if self.heuristica == "admissivel":
            # max_r [ dist(pos, r) + dist(r, saida) ].
            # Admissível: qualquer caminho que colete r e vá à saída custa ao
            # menos isso; o máximo sobre os r pendentes ainda é um lower bound.
            return max(_manhattan(pos, r) + _manhattan(r, saida) for r in restantes)

        if self.heuristica == "guloso":
            # Soma gulosa (vizinho mais próximo) + saída. É a sugerida no SPEC,
            # mas é o comprimento de UM tour viável -> pode SUPERESTIMAR ->
            # NÃO é estritamente admissível -> A* pode devolver caminho subótimo.
            atual = pos
            total = 0
            pendentes = list(restantes)
            while pendentes:
                prox = min(pendentes, key=lambda r: _manhattan(atual, r))
                total += _manhattan(atual, prox)
                atual = prox
                pendentes.remove(prox)
            total += _manhattan(atual, saida)
            return total

        raise ValueError(self.heuristica)  # nunca chega aqui


# ---------------------------------------------------------------------- #
# Autoteste: planeja no mapa seed=42, roda o episódio e compara heurísticas.
# Execute:  python -m agentes.agente_busca
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    env = GridWorldEnv(seed=42)

    print("=== A* com heurística admissível (padrão) ===")
    agente = AgenteBusca(env, heuristica="admissivel")
    metricas = rodar_episodio(env, agente)
    print(f"plano com {len(agente.plano)} ações, {agente.nos_expandidos} nós expandidos")
    print("métricas do episódio:", metricas)

    print("\n=== Comparação de heurísticas (mesmo mapa) ===")
    # Referência de ótimo: Dijkstra (heurística nula) sempre acha o menor custo.
    ref = AgenteBusca(env, heuristica="nula")
    otimo = len(ref.planejar())

    print(f"{'heuristica':<12} {'ações':>6} {'nós exp.':>9}  {'ótimo?':>15}")
    for h in AgenteBusca.HEURISTICAS:
        ag = AgenteBusca(env, heuristica=h)
        plano = ag.planejar()
        n = len(plano) if plano else -1
        marca = "sim" if n == otimo else "NÃO (subótimo)"
        print(f"{h:<12} {n:>6} {ag.nos_expandidos:>9}  {marca:>15}")
    print(
        "\nLeitura: 'nula' (Dijkstra) e 'admissivel'/'ingenua' dão o ótimo;\n"
        "'admissivel' expande MENOS nós que Dijkstra (heurística informada).\n"
        "'guloso' pode aparecer com nº de ações diferente -> subótimo, como previsto."
    )
