"""
Testes do agente de busca A* (seção 3.1).

    python -m unittest tests.test_agente_busca -v
"""

import unittest

from ambiente.grid_world import GridWorldEnv
from agentes.agente_aleatorio import rodar_episodio
from agentes.agente_busca import AgenteBusca, _manhattan


def montar_cenario(N=5, pos_inicial=(0, 0), pos_saida=(4, 4),
                   recursos=None, armadilhas=None, energia_inicial=1000):
    """Cria um GridWorldEnv com layout 100% controlado (ver testes do ambiente)."""
    recursos = list(recursos) if recursos is not None else []
    armadilhas = set(armadilhas) if armadilhas is not None else set()
    env = GridWorldEnv(N=N, n_recursos=len(recursos), n_armadilhas=len(armadilhas),
                       energia_inicial=energia_inicial, seed=0)
    env.pos_inicial = pos_inicial
    env.pos_saida = pos_saida
    env.recursos = recursos
    env.armadilhas = armadilhas
    env.mascara_completa = (1 << len(recursos)) - 1
    env.reset()
    return env


class TestManhattan(unittest.TestCase):
    def test_valor(self):
        self.assertEqual(_manhattan((0, 0), (3, 4)), 7)
        self.assertEqual(_manhattan((2, 2), (2, 2)), 0)


class TestPlanejamentoBasico(unittest.TestCase):

    def test_resolve_mapa_padrao(self):
        env = GridWorldEnv(seed=42)
        agente = AgenteBusca(env)
        m = rodar_episodio(env, agente)
        self.assertTrue(m["sucesso"])
        self.assertEqual(m["motivo"], "saida_sucesso")

    def test_plano_otimo_em_cenario_conhecido(self):
        # (0,0) -> recurso (0,2) -> saída (2,2). Ótimo = 2 + 2 = 4 passos.
        env = montar_cenario(N=3, pos_inicial=(0, 0), pos_saida=(2, 2),
                             recursos=[(0, 2)])
        agente = AgenteBusca(env, heuristica="admissivel")
        plano = agente.planejar()
        self.assertEqual(len(plano), 4)

    def test_plano_executa_com_sucesso(self):
        env = montar_cenario(N=3, pos_inicial=(0, 0), pos_saida=(2, 2),
                             recursos=[(0, 2), (2, 0)])
        agente = AgenteBusca(env)
        m = rodar_episodio(env, agente)
        self.assertTrue(m["sucesso"])


class TestHeuristicas(unittest.TestCase):

    def test_admissivel_e_ingenua_dao_otimo(self):
        # Heurísticas admissíveis devem ter o MESMO custo do Dijkstra (nula).
        env = GridWorldEnv(seed=42)
        otimo = len(AgenteBusca(env, heuristica="nula").planejar())
        for h in ("admissivel", "ingenua"):
            self.assertEqual(len(AgenteBusca(env, heuristica=h).planejar()), otimo,
                             f"heurística {h} deveria ser ótima")

    def test_admissivel_expande_menos_que_dijkstra(self):
        # Heurística informada explora menos nós que a busca cega.
        env = GridWorldEnv(seed=42)
        cega = AgenteBusca(env, heuristica="nula")
        cega.planejar()
        info = AgenteBusca(env, heuristica="admissivel")
        info.planejar()
        self.assertLess(info.nos_expandidos, cega.nos_expandidos)

    def test_heuristica_invalida_levanta_erro(self):
        with self.assertRaises(ValueError):
            AgenteBusca(GridWorldEnv(seed=1), heuristica="inexistente")


class TestArmadilhasEInviabilidade(unittest.TestCase):

    def test_desvia_de_armadilha(self):
        # Caminho reto em y=0 bloqueado por armadilha em (1,0): precisa contornar
        # e mesmo assim NUNCA pisar na armadilha.
        env = montar_cenario(N=3, pos_inicial=(0, 0), pos_saida=(2, 0),
                             recursos=[(2, 2)], armadilhas=[(1, 0)])
        agente = AgenteBusca(env)
        m = rodar_episodio(env, agente)
        self.assertTrue(m["sucesso"])
        self.assertNotEqual(m["motivo"], "armadilha")

    def test_saida_murada_nao_tem_solucao(self):
        # Saída em (1,1) cercada de armadilhas nos 4 vizinhos -> inatingível.
        env = montar_cenario(N=3, pos_inicial=(0, 0), pos_saida=(1, 1),
                             recursos=[], armadilhas=[(0, 1), (2, 1), (1, 0), (1, 2)])
        agente = AgenteBusca(env)
        self.assertIsNone(agente.planejar())


if __name__ == "__main__":
    unittest.main(verbosity=2)
