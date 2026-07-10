"""
agente_aleatorio.py — Agente aleatório (baseline). Seção 3.4 do SPEC.

Escolhe uma ação uniformemente ao acaso a cada passo, ignorando o estado.
Serve como PISO de comparação no protocolo de avaliação (seção 4): qualquer
agente "inteligente" precisa, no mínimo, superar o aleatório.

Define também a INTERFACE que todos os agentes do projeto vão seguir:

    agente.reset()                 # (opcional) reinicia estado interno do agente
    agente.escolher_acao(estado)   # recebe o estado do env e devolve 0..3

e a função utilitária `rodar_episodio`, genérica para qualquer agente.
"""

import random

from ambiente.grid_world import GridWorldEnv


class AgenteAleatorio:
    """Baseline: ação uniforme em {0,1,2,3}, sem olhar o estado."""

    def __init__(self, n_acoes=4, seed=None):
        self.n_acoes = n_acoes
        self.rng = random.Random(seed)

    def reset(self):
        # O agente aleatório não guarda estado entre episódios; nada a fazer.
        pass

    def escolher_acao(self, estado):
        # Ignora o `estado` de propósito — é o que o torna "aleatório".
        return self.rng.randrange(self.n_acoes)


def rodar_episodio(env, agente, render=False):
    """
    Roda UM episódio completo do agente no ambiente.

    Devolve um dicionário com as métricas do protocolo de avaliação (seção 4):
        sucesso        -> bool
        recompensa     -> soma das recompensas do episódio
        passos         -> nº de passos até terminar
        energia_final  -> energia restante ao final
        motivo         -> por que o episódio terminou (info do último step)

    Genérica: funciona com QUALQUER objeto que tenha escolher_acao(estado).
    """
    estado = env.reset()
    if hasattr(agente, "reset"):
        agente.reset()

    recompensa_total = 0.0
    info = {}
    if render:
        env.render()

    while True:
        acao = agente.escolher_acao(estado)
        estado, recompensa, terminado, info = env.step(acao)
        recompensa_total += recompensa
        if render:
            env.render()
        if terminado:
            break

    return {
        "sucesso": info.get("sucesso", False),
        "recompensa": recompensa_total,
        "passos": env.passos,
        "energia_final": env.energia,
        "motivo": info.get("motivo_termino", info.get("motivo")),
    }


# ---------------------------------------------------------------------- #
# Autoteste: 1 episódio detalhado + estatística de 100 episódios.
# Execute:  python -m agentes.agente_aleatorio
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    env = GridWorldEnv(seed=42)
    agente = AgenteAleatorio(n_acoes=env.n_acoes, seed=0)

    print("=== Um episódio (métricas ao final) ===")
    metricas = rodar_episodio(env, agente, render=False)
    print(metricas)

    print("\n=== 100 episódios: como se comporta o baseline ===")
    n = 100
    sucessos = 0
    soma_recompensa = 0.0
    soma_passos = 0
    for _ in range(n):
        m = rodar_episodio(env, agente)
        sucessos += int(m["sucesso"])
        soma_recompensa += m["recompensa"]
        soma_passos += m["passos"]

    print(f"taxa de sucesso : {sucessos}/{n} = {100 * sucessos / n:.1f}%")
    print(f"recompensa média: {soma_recompensa / n:.1f}")
    print(f"passos médios   : {soma_passos / n:.1f}")
    print("\n(esperado: taxa de sucesso ~0% — é o piso que os outros devem superar)")
