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
    python visualizacao.py                          # MENU interativo (escolher tudo na tela)
    python visualizacao.py --menu                   # idem, menu interativo
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


def executar(cfg):
    """
    Roda a animação para a configuração `cfg` (dict) e imprime o resumo final.

    Usada tanto pela linha de comando quanto pelo menu interativo — as duas
    entradas montam o mesmo dicionário de configuração.
    """
    nomes = ["aleatorio", "astar", "qlearning"] if cfg["comparar"] else [cfg["agente"]]

    resultados = []
    for nome in nomes:
        env = GridWorldEnv(seed=cfg["seed"])
        agente, rotulo = _criar_agente(nome, env, cfg["heuristica"], cfg["episodios"])
        m = animar(env, agente, rotulo, delay=cfg["delay"], usar_cor=cfg["cor"],
                   ascii_mode=cfg["ascii"], passo_a_passo=cfg["passo_a_passo"])
        resultados.append((rotulo, m))
        if cfg["comparar"] and nome != nomes[-1]:
            _entrada("\n  [ENTER para ver o próximo agente]")

    print("\n" + "=" * 78)
    print(f"  RESULTADO (mapa seed={cfg['seed']})")
    print("=" * 78)
    for rotulo, m in resultados:
        print(_resumo(rotulo, m))
    print()
    return resultados


# ----------------------------------------------------------------------- #
# Menu interativo
# ----------------------------------------------------------------------- #
def _entrada(prompt=""):
    """input() robusto: Ctrl-C / fim-de-entrada encerram educadamente."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  até mais!")
        sys.exit(0)


def _ler_int(prompt, atual, minimo=None):
    """Lê um inteiro; ENTER vazio mantém o valor atual."""
    while True:
        txt = _entrada(f"{prompt} [{atual}]: ")
        if txt == "":
            return atual
        try:
            v = int(txt)
        except ValueError:
            print("  valor inválido — digite um número inteiro.")
            continue
        if minimo is not None and v < minimo:
            print(f"  precisa ser >= {minimo}.")
            continue
        return v


def _ler_float(prompt, atual, minimo=0.0):
    """Lê um número (aceita vírgula); ENTER vazio mantém o valor atual."""
    while True:
        txt = _entrada(f"{prompt} [{atual}]: ")
        if txt == "":
            return atual
        try:
            v = float(txt.replace(",", "."))
        except ValueError:
            print("  valor inválido — digite um número.")
            continue
        if v < minimo:
            print(f"  precisa ser >= {minimo}.")
            continue
        return v


def _limpar_tela():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def _sim_nao(valor):
    return "sim" if valor else "não"


def _menu_agente(cfg):
    print("\n  Qual agente?")
    print("    [1] Aleatório  (baseline)")
    print("    [2] A*         (busca heurística)")
    print("    [3] Q-Learning (aprendizado por reforço)")
    print("    [4] Comparar os três em sequência")
    esc = _entrada("  escolha: ")
    if esc == "1":
        cfg["comparar"], cfg["agente"] = False, "aleatorio"
    elif esc == "2":
        cfg["comparar"], cfg["agente"] = False, "astar"
    elif esc == "3":
        cfg["comparar"], cfg["agente"] = False, "qlearning"
    elif esc == "4":
        cfg["comparar"] = True


def _menu_heuristica(cfg):
    hs = AgenteBusca.HEURISTICAS
    print("\n  Heurística do A*:")
    for i, h in enumerate(hs, 1):
        print(f"    [{i}] {h}")
    esc = _entrada("  escolha: ")
    if esc.isdigit() and 1 <= int(esc) <= len(hs):
        cfg["heuristica"] = hs[int(esc) - 1]


def _desenhar_menu(cfg):
    _limpar_tela()
    alvo = "comparar os três" if cfg["comparar"] else cfg["agente"]
    print("=" * 52)
    print("  GridWorld — menu interativo")
    print("=" * 52)
    print("  Configuração atual:")
    print(f"    agente      : {alvo}")
    print(f"    seed (mapa) : {cfg['seed']}")
    print(f"    heurística  : {cfg['heuristica']}   (usada só pelo A*)")
    print(f"    episódios   : {cfg['episodios']}   (treino do Q-Learning)")
    print(f"    velocidade  : delay {cfg['delay']}s   |   passo-a-passo: {_sim_nao(cfg['passo_a_passo'])}")
    print(f"    aparência   : ascii {_sim_nao(cfg['ascii'])}   |   cores {_sim_nao(cfg['cor'])}")
    print("-" * 52)
    print("  [1] Agente / comparar        [5] Velocidade (delay)")
    print("  [2] Seed do mapa             [6] Passo-a-passo (liga/desliga)")
    print("  [3] Heurística do A*         [7] ASCII (liga/desliga)")
    print("  [4] Episódios do Q-Learning  [8] Cores (liga/desliga)")
    print("-" * 52)
    print("  [R ou ENTER] ▶ RODAR         [Q] Sair")
    print("=" * 52)


def menu(cfg):
    """Loop do menu: ajusta `cfg`, roda a animação e volta para novas escolhas."""
    while True:
        _desenhar_menu(cfg)
        esc = _entrada("  > ").lower()

        if esc in ("", "r"):
            executar(cfg)
            _entrada("\n  [ENTER para voltar ao menu]")
        elif esc == "q":
            print("  até mais!")
            return
        elif esc == "1":
            _menu_agente(cfg)
        elif esc == "2":
            cfg["seed"] = _ler_int("  nova seed do mapa", cfg["seed"])
        elif esc == "3":
            _menu_heuristica(cfg)
        elif esc == "4":
            cfg["episodios"] = _ler_int("  episódios de treino", cfg["episodios"], minimo=1)
        elif esc == "5":
            cfg["delay"] = _ler_float("  delay em segundos", cfg["delay"])
        elif esc == "6":
            cfg["passo_a_passo"] = not cfg["passo_a_passo"]
        elif esc == "7":
            cfg["ascii"] = not cfg["ascii"]
        elif esc == "8":
            cfg["cor"] = not cfg["cor"]
        # qualquer outra tecla: apenas redesenha o menu


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
    p.add_argument("--menu", action="store_true",
                   help="abre o menu interativo (padrão quando não há argumentos)")
    args = p.parse_args()

    cfg = {
        "agente": args.agente,
        "comparar": args.comparar,
        "seed": args.seed,
        "heuristica": args.heuristica,
        "episodios": args.episodios,
        "delay": args.delay,
        "passo_a_passo": args.passo_a_passo,
        "ascii": args.ascii,
        "cor": not args.sem_cor,
    }

    # Sem nenhum argumento, ou com --menu, abre o menu interativo; caso
    # contrário, roda direto o que foi pedido pela linha de comando.
    if args.menu or len(sys.argv) == 1:
        menu(cfg)
    else:
        executar(cfg)


if __name__ == "__main__":
    main()
