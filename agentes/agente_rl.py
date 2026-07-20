"""
agente_rl.py — Agente de Aprendizado por Reforço (Q-Learning tabular). Seção 3.2.

Diferente do A* (que PLANEJA), este agente APRENDE por tentativa e erro: roda
milhares de episódios no ambiente, ajustando uma Q-table Q[estado][ação] pela
regra do Q-Learning, e ao final segue a política gulosa aprendida.

Formulação computacional (material para o vídeo, item 6):
    - Estado discreto : (posicao, coletados_bitmask)  -- energia É DESCARTADA
    - Ação            : 0..3 (cima/baixo/esquerda/direita)
    - Recompensa      : a mesma do ambiente (seção 2.4)
    - Atualização     : Q(s,a) <- Q(s,a) + α [ r + γ·max_a' Q(s',a') − Q(s,a) ]
    - Exploração      : ε-greedy com decaimento (explora no início, explota no fim)

Por que descartar a energia do estado? (SPEC seção 6, item 1 / ENTREGA_1 seção 5)
Incluí-la multiplicaria o espaço de estados por ~300 (energia_inicial), tornando
a Q-table gigante e lenta para preencher. A energia continua valendo como
mecanismo de TÉRMINO no ambiente; o agente só não a "enxerga" no estado. É uma
simplificação do MDP — documentada como limitação para o vídeo (item 9).

Espaço de estados discreto: posição (N*N) × bitmask (2^n_recursos).
No mapa padrão 10×10 com 5 recursos: 100 × 32 = 3200 estados × 4 ações.
"""

import random
from collections import defaultdict

from ambiente.grid_world import GridWorldEnv
from agentes.agente_aleatorio import rodar_episodio


class AgenteRL:
    """
    Q-Learning tabular com política ε-greedy decrescente. Segue a interface de
    agente do projeto:
        reset()                 -> no-op (NÃO apaga o que foi aprendido)
        escolher_acao(estado)   -> ação GULOSA sobre a Q-table (inferência)

    Treino: chame treinar(n_episodios). O aprendizado fica na Q-table interna.

    Hiperparâmetros (SPEC 3.2, valores iniciais a ajustar empiricamente):
        alpha         (α) 0.10  taxa de aprendizado  — velocidade × estabilidade
        gamma         (γ) 0.95  fator de desconto    — peso das recompensas futuras
        epsilon       (ε) 1.00  exploração inicial
        epsilon_min       0.05  exploração residual mínima
        epsilon_decay     0.995 decaimento de ε por episódio
    """

    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995, seed=None):
        self.env = env
        self.n_acoes = env.n_acoes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_inicial = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)

        # Q-table: estado_discreto -> lista de Q-values, um por ação.
        # defaultdict cria [0, 0, 0, 0] na primeira vez que um estado aparece.
        self.Q = defaultdict(lambda: [0.0] * self.n_acoes)

        # Curva de aprendizado: recompensa acumulada de cada episódio de treino
        # (usada no gráfico exigido pelo item 5/8 do vídeo).
        self.historico_recompensas = []
        self.treinado = False

    # ------------------------------------------------------------------ #
    # Discretização do estado
    # ------------------------------------------------------------------ #
    @staticmethod
    def discretizar(estado):
        """
        Converte o estado do ambiente (pos, coletados, energia) na chave da
        Q-table (pos, coletados) — a energia é DESCARTADA de propósito (ver
        docstring do módulo). Continua sendo uma tupla hashable.
        """
        pos, coletados, _energia = estado
        return (pos, coletados)

    # ------------------------------------------------------------------ #
    # Interface de agente (inferência)
    # ------------------------------------------------------------------ #
    def reset(self):
        # Entre episódios de AVALIAÇÃO não se apaga a Q-table aprendida.
        # (O reset do aprendizado, se desejado, é criar um novo AgenteRL.)
        pass

    def escolher_acao(self, estado):
        """Inferência: ação gulosa (greedy) sobre a política aprendida."""
        return self._melhor_acao(self.discretizar(estado))

    # ------------------------------------------------------------------ #
    # Política
    # ------------------------------------------------------------------ #
    def _melhor_acao(self, s):
        """argmax_a Q(s,a) com desempate aleatório entre ações de valor máximo."""
        q = self.Q[s]
        melhor = max(q)
        candidatos = [a for a, v in enumerate(q) if v == melhor]
        return self.rng.choice(candidatos)

    def _acao_epsilon_greedy(self, s):
        """Com prob. ε explora (ação aleatória); senão explota (melhor ação)."""
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_acoes)
        return self._melhor_acao(s)

    # ------------------------------------------------------------------ #
    # Treino
    # ------------------------------------------------------------------ #
    def treinar(self, n_episodios=5000, log_cada=0):
        """
        Roda n_episodios de Q-Learning, atualizando a Q-table a cada passo.

        Devolve `historico_recompensas` (recompensa acumulada por episódio),
        que é a curva de aprendizado. Se log_cada > 0, imprime um resumo
        parcial a cada `log_cada` episódios (útil para acompanhar a convergência).
        """
        self.historico_recompensas = []

        for ep in range(n_episodios):
            estado = self.env.reset()
            s = self.discretizar(estado)
            recompensa_total = 0.0

            while True:
                a = self._acao_epsilon_greedy(s)
                estado_prox, recompensa, terminado, _info = self.env.step(a)
                s_prox = self.discretizar(estado_prox)
                recompensa_total += recompensa

                # Alvo do Q-Learning (off-policy): usa o MELHOR próximo Q.
                # Em estado terminal não há futuro -> o valor futuro é 0.
                melhor_prox = 0.0 if terminado else max(self.Q[s_prox])
                alvo = recompensa + self.gamma * melhor_prox
                self.Q[s][a] += self.alpha * (alvo - self.Q[s][a])

                s = s_prox
                if terminado:
                    break

            # Decaimento de ε: explora muito no começo, quase nada no fim.
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.historico_recompensas.append(recompensa_total)

            if log_cada and (ep + 1) % log_cada == 0:
                janela = self.historico_recompensas[-log_cada:]
                media = sum(janela) / len(janela)
                print(f"  episódio {ep + 1:>5}/{n_episodios}  "
                      f"ε={self.epsilon:.3f}  "
                      f"recompensa média (últimos {log_cada}): {media:+.1f}")

        self.treinado = True
        return self.historico_recompensas

    @property
    def n_estados_vistos(self):
        """Quantos estados distintos entraram na Q-table (cobertura do aprendizado)."""
        return len(self.Q)


# ---------------------------------------------------------------------- #
# Curva de aprendizado (gráfico exigido no item 5/8 do vídeo)
# ---------------------------------------------------------------------- #
def _media_movel(valores, janela):
    """Média móvel simples de `janela` pontos ([] se não houver pontos suficientes)."""
    if janela <= 1 or len(valores) < janela:
        return []
    medias = []
    soma = sum(valores[:janela])
    medias.append(soma / janela)
    for i in range(janela, len(valores)):
        soma += valores[i] - valores[i - janela]
        medias.append(soma / janela)
    return medias


def salvar_curva_aprendizado(historico,
                             caminho="resultados/graficos/curva_qlearning.png",
                             janela_media=100,
                             titulo="Curva de aprendizado — Q-Learning"):
    """
    Plota a recompensa por episódio + média móvel e salva a figura em `caminho`.

    O matplotlib é importado de forma PREGUIÇOSA (o resto do módulo é stdlib
    puro), então o agente treina e roda mesmo sem matplotlib instalado — só a
    plotagem exige a lib. Devolve o caminho salvo, ou None se a lib faltar.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # backend sem janela: salva direto em arquivo
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [aviso] matplotlib não instalado — curva pulada. "
              "Instale com: pip install matplotlib")
        return None

    import os
    pasta = os.path.dirname(caminho)
    if pasta:
        os.makedirs(pasta, exist_ok=True)

    episodios = range(1, len(historico) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(episodios, historico, linewidth=0.6, alpha=0.35,
             label="recompensa por episódio")

    media = _media_movel(historico, janela_media)
    if media:
        inicio = len(historico) - len(media) + 1
        plt.plot(range(inicio, len(historico) + 1), media, linewidth=2.0,
                 color="crimson", label=f"média móvel ({janela_media} episódios)")

    plt.xlabel("episódio de treino")
    plt.ylabel("recompensa acumulada")
    plt.title(titulo)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=120)
    plt.close()
    return caminho


# ---------------------------------------------------------------------- #
# Autoteste: treina no mapa seed=42, salva a curva e compara com o baseline.
# Execute:  python -m agentes.agente_rl
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    from agentes.agente_aleatorio import AgenteAleatorio

    # Console do Windows costuma ser cp1252 e quebra no "ε"/acentos: força UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    SEED_MAPA = 42
    N_EPISODIOS = 5000

    env = GridWorldEnv(seed=SEED_MAPA)
    agente = AgenteRL(env, seed=0)

    print(f"=== Treinando Q-Learning por {N_EPISODIOS} episódios (mapa seed={SEED_MAPA}) ===")
    agente.treinar(n_episodios=N_EPISODIOS, log_cada=N_EPISODIOS // 10)
    print(f"\nestados distintos na Q-table: {agente.n_estados_vistos}")

    caminho = salvar_curva_aprendizado(agente.historico_recompensas)
    if caminho:
        print(f"curva de aprendizado salva em: {caminho}")

    # Avaliação: política gulosa (ε efetivamente 0) em 100 execuções.
    def taxa_sucesso(ag, n=100):
        sucessos = soma_rec = 0
        for _ in range(n):
            m = rodar_episodio(env, ag)
            sucessos += int(m["sucesso"])
            soma_rec += m["recompensa"]
        return sucessos / n, soma_rec / n

    s_rl, r_rl = taxa_sucesso(agente)
    s_base, r_base = taxa_sucesso(AgenteAleatorio(n_acoes=env.n_acoes, seed=0))

    print("\n=== Q-Learning treinado × baseline aleatório (100 execuções) ===")
    print(f"  {'agente':<22} {'sucesso':>8}  {'recompensa média':>18}")
    print(f"  {'Q-Learning (treinado)':<22} {s_rl:>7.0%}  {r_rl:>18.1f}")
    print(f"  {'Aleatório (baseline)':<22} {s_base:>7.0%}  {r_base:>18.1f}")
