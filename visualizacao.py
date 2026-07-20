"""
visualizacao.py — Animação de um episódio no terminal (estilo "tabuleiro").

Mostra o agente se movendo passo a passo, com:
  - moldura em box-drawing e células com glifos (☺ ◆ ▲ ★);
  - RASTRO do agente (breadcrumbs) — dá para ver o caminho percorrido;
  - barra de energia colorida (verde → amarelo → vermelho conforme esgota);
  - narração dos eventos (coleta, armadilha, sucesso) e recompensa ao vivo.

Serve para entender o comportamento de cada agente e para gravar a "evolução
visual do agente" exigida no vídeo (SPEC seção 8).

Exemplos de uso:
    python visualizacao.py                          # A* no mapa seed=42
    python visualizacao.py --agente aleatorio       # o baseline se atrapalhando
    python visualizacao.py --agente qlearning       # treina e mostra o Q-Learning
    python visualizacao.py --comparar               # aleatório, A* e Q-Learning
    python visualizacao.py --passo-a-passo          # avança com ENTER
    python visualizacao.py --delay 0.4 --seed 7     # mais devagar, outro mapa
    python visualizacao.py --ascii                  # sem glifos/box (terminal simples)
    python visualizacao.py --sem-cor                # sem cores ANSI

NOTA: os glifos e a moldura exigem um terminal com UTF-8 (Windows Terminal já
serve). Se aparecerem "?" ou caixas, use --ascii.
"""

import argparse
import os
import sys
import time

from ambiente.grid_world import GridWorldEnv, NOME_ACAO
from agentes.agente_aleatorio import AgenteAleatorio
from agentes.agente_busca import AgenteBusca
from agentes.agente_rl import AgenteRL


# --- Suporte a ANSI/UTF-8 no Windows ------------------------------------ #
if os.name == "nt":
    os.system("")                      # habilita códigos ANSI no console do Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")   # evita glifos/acentos quebrados
except Exception:
    pass


# --- Glifos por tipo de célula ------------------------------------------ #
#   A = agente · R = recurso pendente · X = armadilha · S = saída
#   o = rastro (célula já visitada) · . = vazio
GLIFOS_UNICODE = {"A": "☺", "R": "◆", "X": "▲", "S": "★", "o": "∘", ".": "·"}
GLIFOS_ASCII = {"A": "A", "R": "R", "X": "X", "S": "S", "o": "o", ".": "."}

CORES = {
    "A": "\033[1;96m",   # ciano brilhante
    "R": "\033[1;93m",   # amarelo
    "X": "\033[1;91m",   # vermelho
    "S": "\033[1;92m",   # verde
    "o": "\033[2;36m",   # ciano apagado (rastro)
    ".": "\033[90m",     # cinza escuro
}
RESET = "\033[0m"
NEGRITO = "\033[1m"
VERDE, AMARELO, VERMELHO = "\033[92m", "\033[93m", "\033[91m"

BOX_UNICODE = dict(tl="╔", tr="╗", bl="╚", br="╝", h="═", v="║", ml="╠", mr="╣")
BOX_ASCII = dict(tl="+", tr="+", bl="+", br="+", h="-", v="|", ml="+", mr="+")


# ----------------------------------------------------------------------- #
# Peças do frame
# ----------------------------------------------------------------------- #
def _tipo_celula(env, pos, rastro, x, y):
    """Decide o que mostrar numa célula (prioridade: agente vence tudo)."""
    cel = (x, y)
    if cel == pos:
        return "A"
    if cel in env.armadilhas:
        return "X"
    if cel == env.pos_saida:
        return "S"
    for i, r in enumerate(env.recursos):
        if cel == r and not (env.coletados & (1 << i)):
            return "R"
    if cel in rastro:
        return "o"
    return "."


def _glifo(tipo, glifos, usar_cor):
    ch = glifos[tipo]
    return f"{CORES[tipo]}{ch}{RESET}" if usar_cor else ch


def _barra_energia(env, usar_cor, ascii_mode, largura=10):
    """Devolve (texto_colorido, largura_visivel) da barra de energia."""
    frac = max(0.0, env.energia / env.energia_inicial)
    cheio = round(frac * largura)
    cheia_ch, vazia_ch = ("#", "-") if ascii_mode else ("█", "░")
    barra = cheia_ch * cheio + vazia_ch * (largura - cheio)
    if not usar_cor:
        return barra, largura
    cor = VERDE if frac > 0.5 else AMARELO if frac > 0.2 else VERMELHO
    return f"{cor}{barra}{RESET}", largura


def _montar_frame(env, rotulo, recompensa_total, evento, rastro, usar_cor, ascii_mode):
    """Monta a string completa de um quadro (moldura + grid + stats)."""
    box = BOX_ASCII if ascii_mode else BOX_UNICODE
    glifos = GLIFOS_ASCII if ascii_mode else GLIFOS_UNICODE
    N = env.N

    # --- linhas do grid (largura visível fixa = 2N-1) ---
    grid = []
    for y in range(N):
        celulas = [_glifo(_tipo_celula(env, env.pos, rastro, x, y), glifos, usar_cor)
                   for x in range(N)]
        grid.append((" ".join(celulas), 2 * N - 1))

    # --- título e stats (guardamos versão colorida + largura visível) ---
    titulo = f"{rotulo}   passo {env.passos:>3}   energia {env.energia:>4}"
    titulo_col = f"{NEGRITO}{titulo}{RESET}" if usar_cor else titulo

    coletados = bin(env.coletados).count("1")
    barra_col, barra_vis = _barra_energia(env, usar_cor, ascii_mode)
    rec_glifo = _glifo("R", glifos, usar_cor)
    stats_col = (f"{rec_glifo} {coletados}/{env.n_recursos}   "
                 f"[{barra_col}]   R:{recompensa_total:+.0f}")
    stats_vis = len(f"X {coletados}/{env.n_recursos}   [") + barra_vis + \
        len(f"]   R:{recompensa_total:+.0f}")

    # --- largura interna da moldura ---
    inner = max(2 * N - 1, len(titulo), stats_vis)

    def linha(conteudo_col, vis):
        return f"{box['v']} {conteudo_col}{' ' * (inner - vis)} {box['v']}"

    barra_h = box["h"] * (inner + 2)
    out = [box["tl"] + barra_h + box["tr"]]
    out.append(linha(titulo_col, len(titulo)))
    out.append(box["ml"] + barra_h + box["mr"])
    for conteudo, vis in grid:
        out.append(linha(conteudo, vis))
    out.append(box["ml"] + barra_h + box["mr"])
    out.append(linha(stats_col, stats_vis))
    out.append(box["bl"] + barra_h + box["br"])

    # --- legenda e narração (fora da moldura) ---
    out.append("")
    out.append(f"  {_glifo('A', glifos, usar_cor)} agente   "
               f"{_glifo('R', glifos, usar_cor)} recurso   "
               f"{_glifo('X', glifos, usar_cor)} armadilha   "
               f"{_glifo('S', glifos, usar_cor)} saída   "
               f"{_glifo('o', glifos, usar_cor)} rastro")
    out.append("")
    out.append(evento)
    return "\n".join(out)


def _descrever_evento(info, acao, recompensa):
    """Traduz o `info` do step numa frase legível (a narração da animação)."""
    motivo = info.get("motivo")
    termino = info.get("motivo_termino")

    if motivo == "coleta":
        msg = f"  >>> COLETOU o recurso {info['recurso']}!  ({recompensa:+.0f})"
    elif motivo == "armadilha":
        msg = f"  >>> CAIU NUMA ARMADILHA — fim do episódio  ({recompensa:+.0f})"
    elif motivo == "saida_sucesso":
        msg = f"  >>> SAÍDA COM TODOS OS RECURSOS — SUCESSO!  ({recompensa:+.0f})"
    elif motivo == "saida_incompleta":
        msg = f"      passou pela saída, mas ainda faltam recursos  ({recompensa:+.0f})"
    else:
        msg = f"      {NOME_ACAO[acao]}  ({recompensa:+.0f})"

    if termino == "exaustao":
        msg += "\n  >>> A ENERGIA ACABOU — fim por exaustão (-20)"
    elif termino == "timeout":
        msg += "\n  >>> ESTOUROU O LIMITE DE PASSOS — fim por timeout (-20)"
    return msg


def _desenhar(frame):
    """Redesenha a partir do topo da tela (sem piscar) e apaga o resto."""
    sys.stdout.write("\033[H" + frame + "\033[J")
    sys.stdout.flush()


# ----------------------------------------------------------------------- #
# Loop de animação
# ----------------------------------------------------------------------- #
def animar(env, agente, rotulo, delay=0.15, usar_cor=True,
           ascii_mode=False, passo_a_passo=False):
    """Roda um episódio animando cada passo. Devolve as métricas finais."""
    estado = env.reset()
    if hasattr(agente, "reset"):
        agente.reset()

    rastro = set()
    recompensa_total = 0.0
    info = {}

    sys.stdout.write("\033[2J")        # limpa a tela uma vez
    _desenhar(_montar_frame(env, rotulo, recompensa_total,
                            "  (início do episódio)", rastro, usar_cor, ascii_mode))
    _pausar(delay, passo_a_passo)

    while True:
        acao = agente.escolher_acao(estado)
        anterior = env.pos                       # célula que vira rastro
        estado, recompensa, terminado, info = env.step(acao)
        rastro.add(anterior)
        recompensa_total += recompensa

        evento = _descrever_evento(info, acao, recompensa)
        _desenhar(_montar_frame(env, rotulo, recompensa_total, evento,
                                rastro, usar_cor, ascii_mode))
        _pausar(delay, passo_a_passo)
        if terminado:
            break

    return {
        "sucesso": info.get("sucesso", False),
        "recompensa": recompensa_total,
        "passos": env.passos,
        "energia_final": env.energia,
        "motivo": info.get("motivo_termino", info.get("motivo")),
    }


def _pausar(delay, passo_a_passo):
    if passo_a_passo:
        try:
            input("\n  [ENTER para o próximo passo]")
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
    else:
        time.sleep(delay)


def _criar_agente(nome, env, heuristica, n_treino=5000):
    if nome == "aleatorio":
        return AgenteAleatorio(n_acoes=env.n_acoes, seed=0), "Aleatório (baseline)"
    if nome == "astar":
        return AgenteBusca(env, heuristica=heuristica), f"A* ({heuristica})"
    if nome == "qlearning":
        agente = AgenteRL(env, seed=0)
        print(f"  treinando Q-Learning por {n_treino} episódios "
              f"(aguarde alguns segundos)...", flush=True)
        agente.treinar(n_episodios=n_treino)
        return agente, "Q-Learning (treinado)"
    raise ValueError(nome)


def _resumo(rotulo, m):
    status = "SUCESSO" if m["sucesso"] else "FALHA  "
    return (f"  {rotulo:<24} {status}  "
            f"recompensa={m['recompensa']:+7.0f}  "
            f"passos={m['passos']:>3}  "
            f"energia_final={m['energia_final']:>3}  "
            f"({m['motivo']})")


def main():
    p = argparse.ArgumentParser(description="Animação do GridWorld no terminal.")
    p.add_argument("--agente", choices=["aleatorio", "astar", "qlearning"],
                   default="astar")
    p.add_argument("--comparar", action="store_true",
                   help="anima aleatório, A* e Q-Learning em sequência e compara")
    p.add_argument("--seed", type=int, default=42, help="mapa a usar (padrão: 42)")
    p.add_argument("--heuristica", default="admissivel",
                   choices=AgenteBusca.HEURISTICAS, help="heurística do A*")
    p.add_argument("--episodios", type=int, default=5000,
                   help="episódios de treino do Q-Learning (padrão: 5000)")
    p.add_argument("--delay", type=float, default=0.15,
                   help="segundos entre passos (padrão: 0.15)")
    p.add_argument("--passo-a-passo", action="store_true",
                   help="avança apertando ENTER em vez de tempo")
    p.add_argument("--ascii", action="store_true",
                   help="usa letras/ASCII em vez de glifos e box-drawing")
    p.add_argument("--sem-cor", action="store_true", help="desliga as cores ANSI")
    args = p.parse_args()

    usar_cor = not args.sem_cor
    nomes = ["aleatorio", "astar", "qlearning"] if args.comparar else [args.agente]

    resultados = []
    for nome in nomes:
        env = GridWorldEnv(seed=args.seed)
        agente, rotulo = _criar_agente(nome, env, args.heuristica, args.episodios)
        m = animar(env, agente, rotulo, delay=args.delay, usar_cor=usar_cor,
                   ascii_mode=args.ascii, passo_a_passo=args.passo_a_passo)
        resultados.append((rotulo, m))
        if args.comparar and nome != nomes[-1]:
            input("\n  [ENTER para ver o próximo agente]")

    print("\n" + "=" * 78)
    print(f"  RESULTADO (mapa seed={args.seed})")
    print("=" * 78)
    for rotulo, m in resultados:
        print(_resumo(rotulo, m))
    print()


if __name__ == "__main__":
    main()
