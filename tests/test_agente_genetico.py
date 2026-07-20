"""
Testes do agente evoluído por Algoritmo Genético (seção 3.3).

    python -m unittest tests.test_agente_genetico -v

As checagens de convergência usam um ambiente TRIVIAL e controlado (grid pequeno,
1 recurso, sem armadilhas) para serem rápidas e determinísticas.
"""

import unittest

from ambiente.grid_world import GridWorldEnv
from agentes.agente_aleatorio import AgenteAleatorio, rodar_episodio
from agentes.agente_genetico import AgenteGenetico


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


class TestGenoma(unittest.TestCase):

    def test_genoma_tem_o_comprimento_certo(self):
        env = montar_cenario()
        agente = AgenteGenetico(env, comprimento=25, seed=0)
        g = agente._genoma_aleatorio()
        self.assertEqual(len(g), 25)
        self.assertTrue(all(a in range(env.n_acoes) for a in g))

    def test_fitness_de_sequencia_conhecida(self):
        # 3x3: (0,0) --baixo--> (0,1) --baixo--> (0,2)=recurso(+10)
        #      --direita--> (1,2) --direita--> (2,2)=saída com tudo (+100)
        # total = -1 +10 -1 +100 = +108
        env = montar_cenario(recursos=[(0, 2)])
        agente = AgenteGenetico(env, seed=0)
        self.assertEqual(agente._fitness([1, 1, 3, 3]), 108.0)


class TestInferencia(unittest.TestCase):

    def test_escolher_acao_segue_o_plano_em_ordem(self):
        env = montar_cenario()
        agente = AgenteGenetico(env, seed=0)
        agente.melhor = [3, 1, 2, 0]
        agente.reset()
        saida = [agente.escolher_acao(None) for _ in range(4)]
        self.assertEqual(saida, [3, 1, 2, 0])

    def test_plano_esgotado_devolve_acao_padrao(self):
        env = montar_cenario()
        agente = AgenteGenetico(env, seed=0)
        agente.melhor = [2]
        agente.reset()
        self.assertEqual(agente.escolher_acao(None), 2)
        self.assertEqual(agente.escolher_acao(None), 0)   # além do plano

    def test_reset_reinicia_o_ponteiro(self):
        env = montar_cenario()
        agente = AgenteGenetico(env, seed=0)
        agente.melhor = [3, 1]
        agente.reset()
        agente.escolher_acao(None)
        agente.reset()
        self.assertEqual(agente.escolher_acao(None), 3)   # voltou ao início

    def test_inferencia_sem_treino_levanta_erro(self):
        env = montar_cenario()
        agente = AgenteGenetico(env, seed=0)
        with self.assertRaises(RuntimeError):
            agente.escolher_acao(None)


class TestEvolucao(unittest.TestCase):

    def test_historico_tem_um_valor_por_geracao(self):
        env = montar_cenario(recursos=[(0, 2)])
        agente = AgenteGenetico(env, comprimento=12, tam_populacao=20,
                                n_geracoes=15, seed=0)
        fmax, fmed = agente.treinar()
        self.assertEqual(len(fmax), 15)
        self.assertEqual(len(fmed), 15)
        self.assertTrue(agente.treinado)

    def test_evolucao_e_deterministica_com_mesma_seed(self):
        env1 = montar_cenario(recursos=[(0, 2)])
        env2 = montar_cenario(recursos=[(0, 2)])
        h1 = AgenteGenetico(env1, comprimento=12, tam_populacao=20,
                            n_geracoes=15, seed=7).treinar()[0]
        h2 = AgenteGenetico(env2, comprimento=12, tam_populacao=20,
                            n_geracoes=15, seed=7).treinar()[0]
        self.assertEqual(h1, h2)

    def test_evolui_para_resolver_ambiente_trivial(self):
        env = montar_cenario(recursos=[(0, 2)])
        agente = AgenteGenetico(env, comprimento=12, tam_populacao=60,
                                n_geracoes=120, seed=0)
        agente.treinar()
        m = rodar_episodio(env, agente)
        self.assertTrue(m["sucesso"])
        self.assertEqual(m["motivo"], "saida_sucesso")

    def test_supera_o_baseline_aleatorio(self):
        env = montar_cenario(recursos=[(0, 2), (2, 0)])
        agente = AgenteGenetico(env, comprimento=16, tam_populacao=80,
                                n_geracoes=150, seed=0)
        agente.treinar()

        def taxa(ag, n=30):
            return sum(rodar_episodio(env, ag)["sucesso"] for _ in range(n)) / n

        self.assertGreater(taxa(agente), taxa(AgenteAleatorio(n_acoes=env.n_acoes, seed=0)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
