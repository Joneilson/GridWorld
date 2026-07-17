# Entrega 1 — Ambiente + Agente Aleatório

> Primeiro entregável do cronograma (SPEC seção 7, janela **10–12/07**):
> *"Ambiente (`grid_world.py`) + agente aleatório rodando ponta a ponta."*

Este documento resume **o que foi pedido, o que foi entregue e como usar**.
A explicação didática de *como cada parte funciona por dentro* está em
[TEORIA.md](TEORIA.md).

---

## 1. O que foi pedido

Do próprio SPEC (seção 9), o pedido inicial:

> "Leia o SPEC.md e crie o scaffold do repositório: comece pelo ambiente
> `grid_world.py` com a classe `GridWorldEnv` (reset, step, render), seguindo
> exatamente as regras da seção 2."

E, na sequência, avançar **um ponto por vez, com validação a cada passo**:
1. Ambiente `GridWorldEnv` (seção 2) — ✅
2. Deixar o ambiente **o mais testado possível** — ✅
3. Verificar **visualmente** o caso de coleta — ✅
4. Agente aleatório (seção 3.4) rodando ponta a ponta — ✅

---

## 2. O que foi entregue

```
GridWorld/
├── SPEC.md                       # especificação de referência (dado)
├── ENTREGA_1.md                  # este arquivo
├── TEORIA.md                     # didática da implementação
├── requirements.txt              # numpy + matplotlib (só nos próximos passos)
├── ambiente/
│   ├── __init__.py
│   └── grid_world.py             # classe GridWorldEnv (reset, step, render)
├── agentes/
│   ├── __init__.py
│   └── agente_aleatorio.py       # AgenteAleatorio + rodar_episodio (baseline)
└── tests/
    ├── test_grid_world.py        # 23 testes das regras da seção 2
    └── test_agente_aleatorio.py  # 5 testes do agente/execução
```

**Ainda não implementado** (próximos passos do SPEC seção 3): agente de busca
A\*, agente Q-Learning, agente genético, protocolo de avaliação e `main.py`.

---

## 3. Como rodar

Nada além da biblioteca padrão do Python é necessário nesta entrega.

```bash
# 1) Ver o AMBIENTE funcionando (passeio aleatório + render)
python -m ambiente.grid_world

# 2) Ver o AGENTE ALEATÓRIO rodando um episódio + estatística de 100 episódios
python -m agentes.agente_aleatorio

# 3) Rodar TODOS os testes
python -m unittest discover -s tests -v

# 4) VER A ANIMAÇÃO no terminal (use este para gravar o vídeo)
python visualizacao.py --comparar          # aleatório e depois A*, com resumo
python visualizacao.py --agente astar      # só o A* resolvendo
python visualizacao.py --passo-a-passo     # avança com ENTER, bom p/ explicar
python visualizacao.py --ascii             # se o terminal não tiver UTF-8 (glifos viram "?")
python visualizacao.py --help              # todas as opções
```

Saída típica do item 2 (o baseline **não** resolve a tarefa — é o piso):

```
taxa de sucesso : 0/100 = 0.0%
recompensa média: -99.7
passos médios   : 59.4
```

---

## 4. Como o ambiente segue a seção 2 (mapa regra → código)

| Regra do SPEC | Onde está no código | Comportamento |
|---|---|---|
| Estado `(pos, bitmask, energia)` (2.2) | `GridWorldEnv._get_estado` | Tupla *hashable* (serve de chave para Q-table) |
| Ações 0–3 cima/baixo/esq/dir (2.3) | dict `ACOES` | `(x,y)`: x=coluna, y=linha, y=0 no topo |
| Movimento inválido na borda (2.3) | `GridWorldEnv.step` | Permanece no lugar, **energia cai mesmo assim** |
| Recompensas (2.4) | `_avaliar_celula` + `step` | passo −1 · coleta +10 · armadilha −50 · sucesso +100 |
| Término (2.5) | flag `self.terminado` | sucesso / armadilha / energia 0 / timeout |
| Energia `3·N·N`, −1/passo (2.6) | `__init__` / `step` | padrão 300 no grid 10×10 |
| Layout fixo por seed (seção 6/3) | `_gerar_layout` | sorteado 1× no `__init__`; `reset` só recoloca o agente |

---

## 5. Decisões fechadas nesta entrega

Duas escolhas em aberto do SPEC (seção 6) foram decididas e **documentadas**:

1. **Exaustão/timeout SOMAM à recompensa do passo** (não sobrescrevem).
   Uma coleta feita no último passo de energia ainda vale `+10 − 20 = −10`.
   *Motivo:* preservar o sinal de aprendizado para RL/GA — chegar à célula do
   recurso continua valendo a pena. (Testado em `test_exaustao_soma_com_coleta`.)

2. **Energia fica no estado do ambiente** (fiel à seção 2.2), mas a decisão de
   **ignorá-la na discretização** de estados fica com cada agente: A\* usa o
   estado completo; Q-Learning/GA vão descartar a energia (usada só como
   término) para não explodir a tabela. Isso será feito ao construir esses
   agentes — nenhuma mudança no ambiente foi necessária.

---

## 6. Estado dos testes

```
Ran 28 tests in 0.002s
OK
```

- **23 testes** cobrem cada linha da tabela de recompensas, movimento na borda,
  determinismo do layout, coleta/dupla-coleta, exaustão, timeout e sucesso.
- **5 testes** cobrem o agente aleatório e a função `rodar_episodio`.

Além disso, o caso de coleta foi **verificado visualmente** (o `R` some do
mapa, o bitmask sobe `00 → 01 → 11`, e a saída fecha em `+100`).

---

## 7. Próximo passo sugerido

Agente de **busca heurística A\*** (SPEC seção 3.1, janela 12–14/07): o primeiro
agente *inteligente*, que já deve superar o baseline com folga.
