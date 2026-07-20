"""
Testes do agente de Aprendizado por Reforço — Q-Learning tabular (seção 3.2).

    python -m unittest tests.test_agente_rl -v

As checagens de convergência usam um ambiente TRIVIAL e controlado (grid pequeno,
1 recurso, sem armadilhas) para serem rápidas e determinísticas — não dependem
do mapa grande seed=42 (que o autoteste de agente_rl.py exercita à mão).
"""

import unittest

from ambiente.grid_world import GridWorldEnv
from agentes.agente_aleatorio import AgenteAleatorio, rodar_episodio
from agentes.agente_rl import AgenteRL


def montar_cenario(N=3, pos_inicial=(0, 0), pos_saida=(2, 2),
                   recursos=None, armadilhas=None, energia_inicial=1000,
                   max_passos=100):
    """Cria um GridWorldEnv com layout 100% controlado (mesmo padrão dos outros testes)."""
    recursos = list(recursos) if recursos is not None else []
    armadilhas = set(armadilhas) if armadilhas is not None else set()
    env = GridWorldEnv(N=N, n_recursos=len(recursos), n_armadilhas=len(armadilhas),
                       energia_inicial=energia_inicial, max_passos=max_passos, seed=0)
    env.pos_inicial = pos_inicial
    env.pos_saida = pos_saida
    env.recursos = recursos
    env.armadilhas = armadilhas
    env.mascara_completa = (1 << len(recursos)) - 1
    env.reset()
    return env


class TestDiscretizacao(unittest.TestCase):

    def test_energia_e_descartada(self):
        # (pos, coletados, energia) -> (pos, coletados). A energia não entra.
        estado = ((3, 4), 0b101, 297)
        self.assertEqual(AgenteRL.discretizar(estado), ((3, 4), 0b101))

    def test_dois_estados_diferindo_so_na_energia_colapsam(self):
        a = AgenteRL.discretizar(((1, 1), 0, 300))
        b = AgenteRL.discretizar(((1, 1), 0, 12))
        self.assertEqual(a, b)


class TestPolitica(unittest.TestCase):

    def test_escolher_acao_e_gulosa(self):
        env = montar_cenario()
        agente = AgenteRL(env, seed=0)
        s = AgenteRL.discretizar(env.reset())
        agente.Q[s] = [0.0, 9.0, 0.0, 0.0]   # ação 1 é claramente a melhor
        self.assertEqual(agente.escolher_acao(env._get_estado()), 1)

    def test_acao_sempre_valida(self):
        env = montar_cenario()
        agente = AgenteRL(env, seed=0)
        estado = env.reset()
        for _ in range(50):
            a = agente.escolher_acao(estado)
            self.assertIn(a, range(env.n_acoes))

    def test_epsilon_greedy_puro_explora(self):
        # Com ε=1.0 toda ação vem do sorteio uniforme -> nunca depende da Q-table.
        env = montar_cenario()
        agente = AgenteRL(env, epsilon=1.0, seed=0)
        s = AgenteRL.discretizar(env.reset())
        agente.Q[s] = [100.0, 0.0, 0.0, 0.0]  # ação 0 seria a "melhor"
        acoes = {agente._acao_epsilon_greedy(s) for _ in range(200)}
        self.assertGreater(len(acoes), 1)     # explorou além da melhor ação


class TestTreino(unittest.TestCase):

    def test_historico_tem_um_valor_por_episodio(self):
        env = montar_cenario()
        agente = AgenteRL(env, seed=0)
        hist = agente.treinar(n_episodios=50)
        self.assertEqual(len(hist), 50)
        self.assertTrue(agente.treinado)

    def test_epsilon_decai_e_respeita_o_minimo(self):
        env = montar_cenario()
        agente = AgenteRL(env, epsilon=1.0, epsilon_min=0.05,
                          epsilon_decay=0.99, seed=0)
        agente.treinar(n_episodios=500)
        self.assertLess(agente.epsilon, 1.0)
        self.assertGreaterEqual(agente.epsilon, 0.05)

    def test_treino_e_deterministico_com_mesma_seed(self):
        env1 = montar_cenario()
        env2 = montar_cenario()
        h1 = AgenteRL(env1, seed=123).treinar(n_episodios=100)
        h2 = AgenteRL(env2, seed=123).treinar(n_episodios=100)
        self.assertEqual(h1, h2)

    def test_preenche_a_qtable(self):
        env = montar_cenario()
        agente = AgenteRL(env, seed=0)
        self.assertEqual(agente.n_estados_vistos, 0)
        agente.treinar(n_episodios=100)
        self.assertGreater(agente.n_estados_vistos, 0)

    def test_aprende_a_resolver_ambiente_trivial(self):
        # (0,0) -> recurso (0,2) -> saída (2,2), sem armadilhas: deve convergir.
        env = montar_cenario(recursos=[(0, 2)])
        agente = AgenteRL(env, seed=0)
        agente.treinar(n_episodios=800)
        m = rodar_episodio(env, agente)   # política gulosa após treino
        self.assertTrue(m["sucesso"])
        self.assertEqual(m["motivo"], "saida_sucesso")

    def test_supera_o_baseline_aleatorio(self):
        env = montar_cenario(recursos=[(0, 2), (2, 0)])
        agente = AgenteRL(env, seed=0)
        agente.treinar(n_episodios=1000)

        def taxa(ag, n=40):
            return sum(rodar_episodio(env, ag)["sucesso"] for _ in range(n)) / n

        s_rl = taxa(agente)
        s_base = taxa(AgenteAleatorio(n_acoes=env.n_acoes, seed=0))
        self.assertGreater(s_rl, s_base)
        self.assertGreaterEqual(s_rl, 0.9)   # treinado resolve quase sempre


if __name__ == "__main__":
    unittest.main(verbosity=2)
