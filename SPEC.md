# SPEC.md — Estudo Dirigido de IA (2026.1)
### Projeto: Agentes Inteligentes em Grid World de Coleta

> Este documento é a especificação de referência do projeto. Serve para orientar o desenvolvimento e também como rascunho-base para o README final e para o roteiro do vídeo.

---

## 1. Contexto escolhido

**Contexto I — Ambiente próprio.** Um grid world 2D de coleta de recursos com energia limitada e obstáculos, no qual três paradigmas de agente serão implementados e comparados:

1. Agente baseado em estado, objetivo e busca heurística (A*)
2. Agente que aprende por reforço (Q-Learning tabular)
3. Agente que aprende por algoritmo genético (evolução de política)

Um agente aleatório será incluído como baseline de referência para o protocolo de avaliação (exigido no item 8 do roteiro do vídeo).

---

## 2. Especificação do ambiente

### 2.1 Cenário
Grid `N x N` (padrão: **10x10**, configurável). Contém:
- `A` — posição do agente (1 por episódio)
- `R` — recursos a coletar (posições fixas ou sorteadas por seed, padrão: 5)
- `X` — armadilhas/obstáculos que terminam o episódio em falha (padrão: 6, fixas por seed)
- `S` — saída (1 posição, só é "válida" depois de coletar todos os recursos — ver regra de término)

### 2.2 Estado (observação)
Vetor/tupla com:
```
estado = (
    posicao_agente: (x, y),
    recursos_coletados: bitmask de tamanho = nº de recursos (ex: 0b00101),
    energia_restante: int
)
```
Para o agente de busca e o de RL, o estado é discreto e enumerável — importante para Q-table (tamanho = `N*N * 2^n_recursos * n_niveis_energia_relevantes`, ou energia pode ser tratada fora do estado do Q-learning e controlada só como condição de término, para não explodir o espaço — **decisão a validar durante implementação**, ver seção 6).

### 2.3 Ações
Discretas, 4 direções:
```
0 = cima, 1 = baixo, 2 = esquerda, 3 = direita
```
Movimento inválido (bateria na borda) = permanece no lugar e ainda consome energia (custo de "tentativa").

### 2.4 Recompensas (para RL e fitness do GA)
| Evento | Recompensa |
|---|---|
| Passo comum (sem coletar nada) | -1 (custo de energia) |
| Coletar um recurso | +10 |
| Colidir com armadilha | -50 (fim de episódio) |
| Chegar à saída com todos os recursos coletados | +100 (fim de episódio, sucesso) |
| Chegar à saída sem todos os recursos coletados | -1 (trata como passo comum; agente não sai) |
| Energia chega a 0 sem sucesso | -20 (fim de episódio, falha por exaustão) |
| Estourar nº máximo de passos (ex: 200) | -20 (fim de episódio, timeout) |

### 2.5 Condições de término
- Sucesso: todos os recursos coletados **e** agente na posição de saída.
- Falha: colisão com armadilha, energia = 0, ou timeout de passos.

### 2.6 Energia
- Energia inicial: parâmetro (padrão: `3 * N*N` — dá folga suficiente para explorar sem ser trivial).
- Decresce 1 por passo (não decresce em outros eventos).

### 2.7 Heurística (para o agente de busca)
Distância de Manhattan até o recurso não coletado mais próximo, somada a uma estimativa de retorno até a saída. Como o problema tem **estado composto** (posição + recursos pendentes), a busca é sobre o grafo de estados, não só sobre o grid físico — isso deve ficar bem claro no vídeo (item 6 do roteiro: formulação computacional).

---

## 3. Especificação dos agentes

### 3.1 Agente de Busca Heurística
- Algoritmo: **A\*** sobre o espaço de estados (posição + bitmask de recursos coletados).
- Heurística admissível sugerida: soma das distâncias de Manhattan para visitar os recursos restantes em ordem gulosa + distância até a saída (não é o TSP ótimo, mas é admissível o suficiente para justificar A* — documentar essa escolha e sua limitação no vídeo).
- Saída: sequência de ações ótima (ou subótima documentada) do estado inicial até o sucesso.
- Sem aprendizado — é planejamento, replanejado do zero a cada execução.

**Parâmetros a expor:**
- Heurística usada (permitir trocar por uma "ingênua", ex: só distância até a saída, para comparação)
- Custo por passo (deve bater com a recompensa do RL para comparação justa)

### 3.2 Agente de Aprendizado por Reforço
- Algoritmo: **Q-Learning tabular** (mais fácil de inspecionar e explicar célula a célula do que uma rede neural — decisão deliberada para o vídeo).
- Política de exploração: ε-greedy com decaimento.
- Q-table indexada por `(estado_discreto, ação)`.

**Hiperparâmetros iniciais (ajustar empiricamente):**
| Parâmetro | Valor inicial | Efeito a discutir no vídeo |
|---|---|---|
| α (taxa de aprendizado) | 0.1 | velocidade vs. estabilidade da convergência |
| γ (fator de desconto) | 0.95 | importância de recompensas futuras (relevante pois a saída só compensa depois de coletar tudo) |
| ε inicial | 1.0 | exploração no início |
| ε decaimento | 0.995 por episódio | transição exploração → exploração |
| ε mínimo | 0.05 | manter exploração residual |
| nº episódios de treino | 5000 (ajustar conforme convergência) | — |

**Métricas a registrar durante treino:** recompensa acumulada por episódio (para o gráfico de curva de aprendizado, exigido no item 5/8 do vídeo).

### 3.3 Agente Genético
- Representação do indivíduo: **política direta** — a forma mais simples e explicável é uma tabela de ação por estado discretizado (mesmo espaço de estados do Q-learning) *ou*, alternativa mais leve, uma sequência fixa de N ações (menos robusta a variações do ambiente, mas mais fácil de visualizar evoluindo). **Recomendação: tabela de política**, por ser mais comparável ao Q-learning.
- Fitness: recompensa total acumulada rodando o indivíduo no ambiente (mesma função de recompensa da seção 2.4).
- Seleção: torneio (tamanho 3, por simplicidade e robustez).
- Crossover: uniforme (cada gene/estado da tabela vem de um dos dois pais com prob. 0.5).
- Mutação: por gene, com probabilidade `p_mut`, sorteia nova ação aleatória.
- Elitismo: manter os 2 melhores indivíduos de cada geração sem alteração.

**Hiperparâmetros iniciais:**
| Parâmetro | Valor inicial |
|---|---|
| Tamanho da população | 100 |
| Nº de gerações | 200 |
| Taxa de mutação | 0.02 |
| Taxa de crossover | 0.7 |
| Tamanho do torneio | 3 |
| Elitismo | 2 indivíduos |

**Métricas a registrar:** fitness máximo e médio por geração (gráfico de evolução).

### 3.4 Agente Aleatório (baseline)
Escolhe ação uniformemente ao acaso a cada passo. Serve apenas como piso de comparação no protocolo de avaliação.

---

## 4. Protocolo de avaliação (item 8 do vídeo)

Para cada agente (heurístico, RL treinado, GA evoluído, aleatório):
- Rodar **30 execuções independentes** (seeds diferentes de posição de recursos/armadilhas, ou seed fixa se o ambiente for determinístico — decidir e documentar).
- Métricas por execução: taxa de sucesso, recompensa total, nº de passos até o término, energia restante ao final.
- Reportar média ± desvio padrão de cada métrica, em tabela comparativa.
- Gráfico de barras ou boxplot comparando os 4 agentes.

---

## 5. Estrutura de repositório

```
projeto-ia/
├── README.md
├── requirements.txt
├── SPEC.md                       (este arquivo)
├── ambiente/
│   ├── __init__.py
│   └── grid_world.py             # classe GridWorldEnv: reset(), step(), render()
├── agentes/
│   ├── __init__.py
│   ├── agente_aleatorio.py
│   ├── agente_busca.py           # A*
│   ├── agente_rl.py              # Q-learning (treino + inferência)
│   └── agente_genetico.py        # GA (evolução + melhor indivíduo)
├── avaliacao/
│   └── comparar_agentes.py       # roda protocolo da seção 4, gera tabela + gráficos
├── resultados/
│   ├── graficos/                 # curvas de aprendizado, evolução do GA, comparativos
│   └── logs/
├── main.py                       # CLI simples: escolher agente e rodar/treinar/avaliar
└── notebooks/                    # opcional, para exploração rápida
```

---

## 6. Decisões em aberto (para resolver durante a implementação)

Estas são escolhas que fazem diferença teórica e devem ser fechadas cedo, documentando o porquê:

1. **Energia entra no estado do Q-learning/GA ou só controla o término do episódio?** Incluir explode o espaço de estados; excluir simplifica mas o agente "não sabe" quanta energia tem. Recomendação inicial: **excluir do estado**, manter só como mecanismo de término — mais simples de explicar e ainda válido teoricamente (é uma simplificação do MDP, documentar como limitação no item 9 do vídeo).
2. **Tamanho do grid**: 10x10 dá um espaço de estados de `100 * 32 = 3200` (para 5 recursos) — tratável para Q-table. Se crescer o grid ou o nº de recursos, reavaliar viabilidade do tabular.
3. **Determinismo do ambiente**: decidir se posições de recursos/armadilhas são fixas (mesma disposição sempre) ou sorteadas por seed a cada episódio de treino. Fixo facilita comparação justa entre os 3 agentes; sorteado testa generalização. Recomendação: **fixo durante treino/comparação principal**, com um experimento extra opcional de generalização (seeds variadas) se sobrar tempo.

---

## 7. Cronograma de referência

| Data | Entregável |
|---|---|
| 10–12/07 | Ambiente (`grid_world.py`) + agente aleatório rodando ponta a ponta |
| 12–14/07 | Agente de busca heurística (A*) funcionando |
| 14–17/07 | Agente de RL treinado, curva de aprendizado plotada |
| **17/07** | **Checkpoint obrigatório**: registrar progresso no ambiente virtual da disciplina |
| 17–20/07 | Agente genético evoluído, curva de fitness plotada |
| 20–22/07 | Avaliação comparativa (seção 4) completa |
| 22–23/07 | README final + roteiro do vídeo |
| 23–24/07 | Gravação, revisão, envio |
| **24/07** | **Entrega final**: vídeo + repositório + requirements.txt |

---

## 8. Checklist de conteúdo obrigatório do vídeo (para não esquecer nada)

- [ ] Funcionamento do ambiente (regras, estados, observações, ações, término)
- [ ] Formato técnico agente↔ambiente (entradas, saídas, ações, recompensas)
- [ ] Funcionamento de cada agente/treinamento à luz da teoria
- [ ] Evolução visual do agente (comportamento inicial → falhas → melhorias)
- [ ] Formulação computacional (estado, ação, objetivo, recompensa, aptidão, custo, heurística)
- [ ] Hiperparâmetros e seus efeitos
- [ ] Protocolo de avaliação, métricas, nº de execuções, comparação com baseline
- [ ] Limitações, falhas, melhorias possíveis, reprodutibilidade
- [ ] Resultado final do agente executando a tarefa

---

## 9. Como retomar este projeto no Claude Code

Ao abrir o Claude Code na pasta do projeto, apontar para este `SPEC.md` e pedir, por exemplo:

> "Leia o SPEC.md e crie o scaffold do repositório: comece pelo ambiente `grid_world.py` com a classe `GridWorldEnv` (reset, step, render), seguindo exatamente as regras da seção 2."

Depois, avançar agente por agente (seção 3), validando cada um antes de seguir para o próximo — evita acumular bugs difíceis de rastrear depois.