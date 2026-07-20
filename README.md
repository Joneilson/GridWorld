# GridWorld — Agentes Inteligentes em um Mundo de Coleta

Estudo dirigido de Inteligência Artificial (2026.1). Um ambiente **Grid World 2D
de coleta de recursos** com energia limitada e armadilhas, no qual diferentes
paradigmas de agente são implementados e comparados sob o mesmo protocolo de
avaliação.

O agente precisa **coletar todos os recursos** espalhados pelo grid e depois
**chegar à saída**, sem cair em armadilhas e sem esgotar a energia.

## Agentes

| Agente | Paradigma | Arquivo | Status |
|---|---|---|---|
| **Aleatório** | Baseline (piso de comparação) | [agentes/agente_aleatorio.py](agentes/agente_aleatorio.py) | ✅ |
| **Busca A\*** | Planejamento / busca heurística | [agentes/agente_busca.py](agentes/agente_busca.py) | ✅ |
| **Q-Learning** | Aprendizado por reforço (tabular) | [agentes/agente_rl.py](agentes/agente_rl.py) | ✅ |
| **Genético** | Evolução de política | [agentes/agente_genetico.py](agentes/agente_genetico.py) | ✅ |

Ainda pendentes: protocolo de avaliação comparativa (30 execuções, tabela +
boxplot) e uma CLI `main.py` unificada.

## O ambiente em resumo

Grid `N×N` (padrão **10×10**), com layout **fixo por seed** (mesma disposição
sempre, para comparação justa entre agentes). Definido em
[ambiente/grid_world.py](ambiente/grid_world.py).

- **Estado:** `(posição (x,y), recursos_coletados (bitmask), energia_restante)` — tupla *hashable*.
- **Ações:** `0=cima, 1=baixo, 2=esquerda, 3=direita`. Bater na borda mantém a posição mas **consome energia**.
- **Recompensas:** passo −1 · coletar recurso +10 · armadilha −50 (fim) · saída com tudo coletado +100 (sucesso) · exaustão/timeout −20 (fim).
- **Término:** sucesso (tudo coletado **e** na saída), armadilha, energia 0, ou timeout de passos.
- **Energia inicial:** `3·N·N` (=300 no 10×10), −1 por passo.

Decisões de modelagem relevantes (documentadas no código):
- **A\*** planeja sobre o espaço de estados composto `(posição × bitmask de recursos)`, tratando armadilhas como paredes; a energia é folgada e não entra na busca.
- **Q-Learning** discretiza o estado como `(posição, bitmask)` e **descarta a energia** para não explodir a Q-table — a energia segue valendo apenas como condição de término (simplificação do MDP, assumida como limitação).
- **Genético** usa o indivíduo como **sequência de ações** (plano de malha aberta), não a tabela de política. A tabela por estado, testada primeiro, estagnava num ótimo local e não resolvia a tarefa; a sequência otimiza a trajetória inteira e resolve o mapa. Trade-off: o plano superajusta a um mapa fixo (re-evolui para outro mapa).

## Estrutura

```
GridWorld/
├── README.md
├── requirements.txt
├── ambiente/
│   └── grid_world.py          # GridWorldEnv: reset(), step(), render()
├── agentes/
│   ├── agente_aleatorio.py    # baseline + rodar_episodio (interface comum)
│   ├── agente_busca.py        # A* sobre (posição, bitmask)
│   ├── agente_rl.py           # Q-Learning tabular (treino + inferência + curva)
│   └── agente_genetico.py     # GA: evolução de sequência de ações + curva de fitness
├── tests/                     # 58 testes (unittest)
├── visualizacao.py            # animação de um episódio no terminal
└── resultados/graficos/       # curvas geradas (não versionado)
```

## Instalação

```bash
pip install -r requirements.txt
```

O ambiente e os agentes são **Python puro** (biblioteca padrão). `numpy` e
`matplotlib` só são necessários para os gráficos (curva de aprendizado / avaliação).

## Como rodar

```bash
# Ambiente funcionando ponta a ponta (passeio aleatório + render)
python -m ambiente.grid_world

# Baseline aleatório: 1 episódio + estatística de 100 (taxa de sucesso ~0%)
python -m agentes.agente_aleatorio

# A* resolvendo o mapa + comparação de heurísticas
python -m agentes.agente_busca

# Treinar o Q-Learning (mapa seed=42), salvar a curva e comparar com o baseline
python -m agentes.agente_rl

# Evoluir o Genético (mapa seed=42), salvar a curva de fitness e comparar
python -m agentes.agente_genetico

# Testes (58)
python -m unittest discover -s tests -v
```

### Visualização no terminal (para o vídeo)

A forma mais fácil é o **menu interativo** — escolha o agente, troque a seed do
mapa, ajuste a velocidade etc. na própria tela, rode e volte ao menu para testar
outra seed sem reiniciar:

```bash
python visualizacao.py            # abre o menu interativo (padrão sem argumentos)
python visualizacao.py --menu     # idem
```

Também dá para ir direto pela linha de comando:

```bash
python visualizacao.py --agente qlearning   # treina e mostra o Q-Learning resolvendo
python visualizacao.py --agente genetico     # evolui e mostra o Genético resolvendo
python visualizacao.py --agente astar        # só o A*
python visualizacao.py --comparar            # aleatório, A*, Q-Learning e Genético
python visualizacao.py --seed 7              # outro mapa
python visualizacao.py --passo-a-passo       # avança com ENTER (bom p/ explicar)
python visualizacao.py --ascii               # se o terminal não tiver UTF-8
python visualizacao.py --help                # todas as opções
```

## Resultado atual (mapa `seed=42`)

| Agente | Taxa de sucesso | Recompensa |
|---|---|---|
| Q-Learning (treinado, 5000 episódios) | **100%** | +124 |
| Genético (evoluído, 500 gerações) | **100%** | +113 |
| Aleatório (baseline) | 0% | −101 |

- O **Q-Learning** parte de recompensa ~−100 e, após ~1000 episódios, estabiliza acima de +100 — curva em `resultados/graficos/curva_qlearning.png`.
- O **Genético** fica preso num ótimo local (~−35) até romper por volta da geração 350 e subir a +113 — curva de fitness em `resultados/graficos/curva_genetico.png`. Esse "salto" é ótimo para mostrar a evolução no vídeo.

## Reprodutibilidade

Tudo é determinístico por seed: o layout do mapa (`GridWorldEnv(seed=...)`) e a
aleatoriedade dos agentes (`seed=...`). Os mesmos comandos acima reproduzem os
mesmos resultados.
