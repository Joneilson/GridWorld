"""
Suíte de testes do ambiente GridWorldEnv.

Cobre TODAS as regras da seção 2 do SPEC.md, uma a uma. Rode com:

    python -m unittest tests.test_grid_world -v

Estratégia: para testar eventos específicos (coleta, armadilha, saída,
exaustão, timeout) precisamos controlar as posições. Como o layout normal é
sorteado por seed, cada teste constrói um cenário PEQUENO e determinístico
sobrescrevendo diretamente os atributos de layout do ambiente após o __init__.
"""

import unittest

from ambiente.grid_world import GridWorldEnv, ACOES


# Índices de ação legíveis nos testes.
CIMA, BAIXO, ESQUERDA, DIREITA = 0, 1, 2, 3


def montar_cenario(
    N=5,
    pos_inicial=(0, 0),
    pos_saida=(4, 4),
    recursos=None,
    armadilhas=None,
    energia_inicial=100,
    max_passos=200,
):
    """
    Cria um GridWorldEnv com layout 100% controlado (ignora o sorteio).

    Retorna o ambiente já com reset() aplicado.
    """
    recursos = list(recursos) if recursos is not None else []
    armadilhas = set(armadilhas) if armadilhas is not None else set()

    env = GridWorldEnv(
        N=N,
        n_recursos=len(recursos),
        n_armadilhas=len(armadilhas),
        energia_inicial=energia_inicial,
        max_passos=max_passos,
        seed=0,
    )
    # Sobrescreve o layout sorteado por um layout determinístico do teste.
    env.pos_inicial = pos_inicial
    env.pos_saida = pos_saida
    env.recursos = recursos
    env.armadilhas = armadilhas
    env.mascara_completa = (1 << len(recursos)) - 1
    env.reset()
    return env


class TestLayoutESeed(unittest.TestCase):
    """Seção 2.1 e seção 6/3: layout válido e fixo por seed."""

    def test_mesma_seed_mesmo_layout(self):
        a = GridWorldEnv(seed=123)
        b = GridWorldEnv(seed=123)
        self.assertEqual(a.pos_inicial, b.pos_inicial)
        self.assertEqual(a.pos_saida, b.pos_saida)
        self.assertEqual(a.recursos, b.recursos)
        self.assertEqual(a.armadilhas, b.armadilhas)

    def test_layout_fixo_entre_resets(self):
        env = GridWorldEnv(seed=7)
        antes = (env.pos_inicial, env.pos_saida, list(env.recursos), set(env.armadilhas))
        env.reset()
        env.step(DIREITA)
        env.reset()
        depois = (env.pos_inicial, env.pos_saida, list(env.recursos), set(env.armadilhas))
        self.assertEqual(antes, depois)

    def test_todas_as_posicoes_sao_distintas(self):
        env = GridWorldEnv(seed=99)
        celulas = [env.pos_inicial, env.pos_saida, *env.recursos, *env.armadilhas]
        self.assertEqual(len(celulas), len(set(celulas)))

    def test_cenario_impossivel_levanta_erro(self):
        with self.assertRaises(ValueError):
            # 2x2 = 4 células, mas 2 + 3 recursos + 3 armadilhas = 8 > 4.
            GridWorldEnv(N=2, n_recursos=3, n_armadilhas=3, seed=1)


class TestEstadoEReset(unittest.TestCase):
    """Seção 2.2: formato do estado; reset restaura tudo."""

    def test_formato_do_estado(self):
        env = montar_cenario()
        estado = env.reset()
        pos, bitmask, energia = estado
        self.assertEqual(pos, (0, 0))
        self.assertEqual(bitmask, 0)
        self.assertEqual(energia, 100)
        self.assertIsInstance(estado, tuple)

    def test_energia_inicial_padrao_3NN(self):
        env = GridWorldEnv(N=10, seed=1)
        env.reset()
        self.assertEqual(env.energia, 3 * 10 * 10)

    def test_reset_restaura_apos_episodio(self):
        env = montar_cenario()
        env.step(DIREITA)
        env.step(BAIXO)
        env.reset()
        self.assertEqual(env.pos, env.pos_inicial)
        self.assertEqual(env.coletados, 0)
        self.assertEqual(env.energia, env.energia_inicial)
        self.assertEqual(env.passos, 0)
        self.assertFalse(env.terminado)


class TestMovimento(unittest.TestCase):
    """Seção 2.3: ações e movimento inválido na borda."""

    def test_quatro_direcoes(self):
        env = montar_cenario(pos_inicial=(2, 2))
        env.step(DIREITA)
        self.assertEqual(env.pos, (3, 2))
        env.step(BAIXO)
        self.assertEqual(env.pos, (3, 3))
        env.step(ESQUERDA)
        self.assertEqual(env.pos, (2, 3))
        env.step(CIMA)
        self.assertEqual(env.pos, (2, 2))

    def test_movimento_invalido_permanece_mas_gasta_energia(self):
        env = montar_cenario(pos_inicial=(0, 0))
        estado, recompensa, terminado, info = env.step(CIMA)  # y = -1, inválido
        self.assertEqual(env.pos, (0, 0))          # permaneceu no lugar
        self.assertEqual(env.energia, 99)          # mas gastou energia
        self.assertEqual(env.passos, 1)
        self.assertEqual(recompensa, -1.0)         # custo de passo comum

    def test_acao_invalida_levanta_erro(self):
        env = montar_cenario()
        with self.assertRaises(ValueError):
            env.step(4)

    def test_energia_decresce_1_por_passo(self):
        env = montar_cenario(pos_inicial=(0, 0), energia_inicial=10)
        for esperado in range(9, 4, -1):
            env.step(DIREITA)
            self.assertEqual(env.energia, esperado)


class TestRecompensas(unittest.TestCase):
    """Seção 2.4: cada linha da tabela de recompensas."""

    def test_passo_comum(self):
        env = montar_cenario(pos_inicial=(0, 0))
        _, r, term, info = env.step(DIREITA)
        self.assertEqual(r, -1.0)
        self.assertFalse(term)
        self.assertEqual(info["motivo"], "passo")

    def test_coleta_recurso(self):
        env = montar_cenario(pos_inicial=(0, 0), recursos=[(1, 0)])
        _, r, term, info = env.step(DIREITA)  # entra em (1,0)
        self.assertEqual(r, +10.0)
        self.assertFalse(term)
        self.assertEqual(env.coletados, 0b1)
        self.assertEqual(info["motivo"], "coleta")

    def test_nao_coleta_recurso_duas_vezes(self):
        env = montar_cenario(pos_inicial=(0, 0), recursos=[(1, 0)])
        env.step(DIREITA)          # coleta (1,0) -> +10
        env.step(ESQUERDA)         # volta para (0,0)
        _, r, _, info = env.step(DIREITA)  # volta a (1,0), já coletado
        self.assertEqual(r, -1.0)
        self.assertEqual(env.coletados, 0b1)  # bitmask não muda
        self.assertEqual(info["motivo"], "passo")

    def test_colisao_armadilha_termina_em_falha(self):
        env = montar_cenario(pos_inicial=(0, 0), armadilhas=[(1, 0)])
        _, r, term, info = env.step(DIREITA)
        self.assertEqual(r, -50.0)
        self.assertTrue(term)
        self.assertFalse(info["sucesso"])
        self.assertEqual(info["motivo"], "armadilha")

    def test_saida_sem_todos_recursos_e_passo_comum(self):
        # Saída em (1,0); agente chega lá SEM ter coletado o recurso.
        env = montar_cenario(pos_inicial=(0, 0), pos_saida=(1, 0), recursos=[(4, 4)])
        _, r, term, info = env.step(DIREITA)
        self.assertEqual(r, -1.0)
        self.assertFalse(term)                    # não sai
        self.assertEqual(info["motivo"], "saida_incompleta")

    def test_saida_com_todos_recursos_e_sucesso(self):
        # Recurso em (1,0), saída em (2,0). Coleta e depois sai.
        env = montar_cenario(pos_inicial=(0, 0), pos_saida=(2, 0), recursos=[(1, 0)])
        env.step(DIREITA)                          # coleta em (1,0)
        _, r, term, info = env.step(DIREITA)       # entra na saída (2,0)
        self.assertEqual(r, +100.0)
        self.assertTrue(term)
        self.assertTrue(info["sucesso"])
        self.assertEqual(info["motivo"], "saida_sucesso")


class TestTerminoPorEnergiaETimeout(unittest.TestCase):
    """Seção 2.4/2.5: exaustão e timeout, incluindo a SOMA de recompensas."""

    def test_exaustao_passo_comum(self):
        # Energia 1: um passo comum zera a energia -> -1 + (-20) = -21.
        env = montar_cenario(pos_inicial=(0, 0), energia_inicial=1)
        _, r, term, info = env.step(DIREITA)
        self.assertEqual(r, -21.0)
        self.assertTrue(term)
        self.assertEqual(info["motivo_termino"], "exaustao")
        self.assertFalse(info["sucesso"])

    def test_exaustao_soma_com_coleta(self):
        # Coleta (+10) exatamente quando a energia zera -> +10 - 20 = -10.
        env = montar_cenario(pos_inicial=(0, 0), recursos=[(1, 0)], energia_inicial=1)
        _, r, term, info = env.step(DIREITA)
        self.assertEqual(r, -10.0)
        self.assertTrue(term)
        self.assertEqual(env.coletados, 0b1)       # a coleta CONTOU
        self.assertEqual(info["motivo"], "coleta")
        self.assertEqual(info["motivo_termino"], "exaustao")

    def test_timeout(self):
        # max_passos = 1: o primeiro passo já estoura -> -1 + (-20) = -21.
        env = montar_cenario(pos_inicial=(0, 0), energia_inicial=100, max_passos=1)
        _, r, term, info = env.step(DIREITA)
        self.assertEqual(r, -21.0)
        self.assertTrue(term)
        self.assertEqual(info["motivo_termino"], "timeout")

    def test_sucesso_no_ultimo_passo_nao_vira_timeout(self):
        # Sucesso é terminal e NÃO deve receber a penalidade de timeout por cima.
        env = montar_cenario(
            pos_inicial=(0, 0), pos_saida=(2, 0), recursos=[(1, 0)], max_passos=2
        )
        env.step(DIREITA)                          # passo 1: coleta
        _, r, term, info = env.step(DIREITA)       # passo 2: sai (passos == max)
        self.assertEqual(r, +100.0)                # +100 puro, sem -20
        self.assertTrue(info["sucesso"])


class TestGuardaERender(unittest.TestCase):
    """Robustez: passo após término e saída do render."""

    def test_step_apos_termino_nao_altera_nada(self):
        env = montar_cenario(pos_inicial=(0, 0), armadilhas=[(1, 0)])
        env.step(DIREITA)                          # cai na armadilha, termina
        estado_apos, r, term, info = env.step(DIREITA)
        self.assertTrue(term)
        self.assertEqual(r, 0.0)
        self.assertIn("aviso", info)

    def test_render_contem_agente_e_dimensao_certa(self):
        env = montar_cenario(N=5, pos_inicial=(0, 0))
        mapa = env.render(mode="silencioso")       # qualquer mode != "human" não imprime
        linhas = mapa.split("\n")
        # 1 cabeçalho + 5 linhas de grid.
        self.assertEqual(len(linhas), 1 + 5)
        self.assertIn("A", mapa)


if __name__ == "__main__":
    unittest.main(verbosity=2)
