"""
Testes do agente aleatório e da função rodar_episodio (seção 3.4).

    python -m unittest tests.test_agente_aleatorio -v
"""

import unittest

from ambiente.grid_world import GridWorldEnv
from agentes.agente_aleatorio import AgenteAleatorio, rodar_episodio


class TestAgenteAleatorio(unittest.TestCase):

    def test_acoes_sempre_validas(self):
        agente = AgenteAleatorio(n_acoes=4, seed=1)
        for _ in range(1000):
            acao = agente.escolher_acao(estado=None)
            self.assertIn(acao, {0, 1, 2, 3})

    def test_reprodutivel_por_seed(self):
        a = AgenteAleatorio(seed=123)
        b = AgenteAleatorio(seed=123)
        seq_a = [a.escolher_acao(None) for _ in range(50)]
        seq_b = [b.escolher_acao(None) for _ in range(50)]
        self.assertEqual(seq_a, seq_b)


class TestRodarEpisodio(unittest.TestCase):

    def test_retorna_todas_as_metricas(self):
        env = GridWorldEnv(seed=42)
        agente = AgenteAleatorio(seed=0)
        m = rodar_episodio(env, agente)
        for chave in ("sucesso", "recompensa", "passos", "energia_final", "motivo"):
            self.assertIn(chave, m)
        self.assertIsInstance(m["sucesso"], bool)

    def test_episodio_sempre_termina(self):
        # Com energia limitada e/ou timeout, nenhum episódio pode rodar para sempre.
        env = GridWorldEnv(seed=7, max_passos=200)
        agente = AgenteAleatorio(seed=3)
        m = rodar_episodio(env, agente)
        self.assertLessEqual(m["passos"], 200)

    def test_agente_de_sucesso_e_detectado(self):
        # Cenário forçado: caminho reto que coleta e sai -> rodar_episodio
        # deve reportar sucesso=True. Usa um agente "scriptado".
        env = GridWorldEnv(N=5, n_recursos=1, n_armadilhas=0, energia_inicial=100, seed=0)
        env.pos_inicial = (0, 0)
        env.pos_saida = (2, 0)
        env.recursos = [(1, 0)]
        env.armadilhas = set()
        env.mascara_completa = 1

        class SempreDireita:
            def escolher_acao(self, estado):
                return 3

        m = rodar_episodio(env, SempreDireita())
        self.assertTrue(m["sucesso"])
        self.assertEqual(m["recompensa"], 10.0 + 100.0)
        self.assertEqual(m["passos"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
