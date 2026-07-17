# TEORIA.md — Como o ambiente foi implementado (e como mexer nele sozinho)

Este documento explica **por dentro** como o `GridWorldEnv` foi construído, com
foco em você conseguir **fazer alterações com suas próprias mãos** sem depender
de ajuda. Cada seção termina com um bloco **"➜ Como mexer"** mostrando o ponto
exato do código a tocar.

Arquivo principal: [ambiente/grid_world.py](ambiente/grid_world.py).

---

## 1. A ideia central: um MDP discreto

O ambiente é um **Processo de Decisão de Markov (MDP)**, a formulação padrão de
problemas de agente. Ele tem quatro peças:

| Peça do MDP | No nosso ambiente |
|---|---|
| **Estado** *(onde estou)* | `(posição, recursos_coletados, energia)` |
| **Ação** *(o que faço)* | `0..3` (cima/baixo/esquerda/direita) |
| **Transição** *(o que acontece)* | `step()` move o agente e aplica as regras da célula |
| **Recompensa** *(quão bom foi)* | número devolvido por `step()` |

O agente **não** enxerga o mapa inteiro. Ele só recebe o **estado** e devolve
uma **ação**. Toda a lógica do mundo vive dentro de `step()`. Essa separação é o
que permite plugar qualquer agente (aleatório, A\*, RL, GA) no mesmo ambiente.

O ciclo é sempre:

```
estado = env.reset()
while not terminado:
    acao = agente.escolher_acao(estado)      # decisão do agente
    estado, recompensa, terminado, info = env.step(acao)   # o mundo responde
```

---

## 2. Sistema de coordenadas — a decisão que confunde todo mundo

O SPEC pede a posição como `(x, y)`. Adotamos:

- **`x` = coluna** → 0 na esquerda, cresce para a **direita**
- **`y` = linha**  → 0 no **topo**, cresce para **baixo**

Isso é importante porque define as ações. Cada ação é um deslocamento `(dx, dy)`
no dicionário `ACOES`:

```python
ACOES = {
    0: (0, -1),   # cima     -> y diminui (sobe na tela)
    1: (0, +1),   # baixo    -> y aumenta (desce na tela)
    2: (-1, 0),   # esquerda -> x diminui
    3: (+1, 0),   # direita  -> x aumenta
}
```

⚠️ A pegadinha clássica: "cima" **diminui** `y` porque a linha 0 é o topo. Se
você inverter isso, o agente anda de cabeça para baixo e nada mais bate.

O `render()` respeita essa convenção: ele desenha `y` de 0 (primeira linha
impressa) até `N-1` (última), e dentro de cada linha `x` da esquerda para a
direita.

> **➜ Como mexer:** para adicionar movimento diagonal, adicione entradas em
> `ACOES` (ex.: `4: (+1, +1)`). O resto do código usa `len(ACOES)` e itera sobre
> o dicionário, então quase tudo se ajusta sozinho — só reveja `n_acoes`.

---

## 3. O estado e o truque do *bitmask*

O estado é uma **tupla**:

```python
(posicao, recursos_coletados, energia)
# ex: ((3, 8), 0b01010, 297)
```

Por que tupla? Porque tupla é **hashable** — pode ser usada direto como chave de
dicionário. Isso será essencial para a Q-table do Q-Learning
(`Q[estado][acao]`).

### O que é o bitmask de recursos

Em vez de guardar uma *lista* de quais recursos foram coletados, guardamos **um
único inteiro** onde cada **bit** representa um recurso:

```
recursos = [(1,0), (3,0), (5,5), (7,2), (9,9)]   # 5 recursos, índices 0..4
                bit 0  bit 1  bit 2  bit 3  bit 4

coletados = 0b00101  ->  recursos 0 e 2 coletados; 1, 3, 4 pendentes
```

Operações usadas no código:

| Objetivo | Código | Explicação |
|---|---|---|
| Recurso `i` já coletado? | `coletados & (1 << i)` | isola o bit `i`; ≠0 se ligado |
| Marcar recurso `i` como coletado | `coletados \|= (1 << i)` | liga o bit `i` |
| Coletou **todos**? | `coletados == (1 << n) - 1` | `(1<<n)-1` = `0b111...1` (n uns) |

Por que bitmask e não lista? Porque o estado precisa ser **enumerável e
compacto** para o Q-Learning. Com 5 recursos há só `2^5 = 32` combinações
possíveis de coleta — um número pequeno, controlável. Uma lista não seria
hashable e explodiria a representação.

> **➜ Como mexer:** para mudar o nº de recursos, use `n_recursos=` no construtor.
> A `mascara_completa` (`(1 << n_recursos) - 1`) se recalcula sozinha. Só lembre:
> `2^n_recursos` entra no tamanho do espaço de estados do RL — 5 é confortável,
> 15 já seriam 32768 combinações.

---

## 4. O layout: fixo por seed

O mapa (posição inicial, saída, 5 recursos, 6 armadilhas) é **sorteado uma única
vez** dentro de `_gerar_layout()`, no `__init__`. O `reset()` **não** re-sorteia
— ele só recoloca o agente no início e restaura energia/coletados/passos.

Como o sorteio funciona:

```python
rng = random.Random(seed)                 # gerador determinístico
todas = [(x, y) for y in range(N) for x in range(N)]   # todas as células
rng.shuffle(todas)                        # embaralha
pos_inicial = todas.pop()                 # consome sem repetir ->
pos_saida   = todas.pop()                 # ... garante posições DISTINTAS
recursos    = [todas.pop() for _ in range(n_recursos)]
armadilhas  = set(todas.pop() for _ in range(n_armadilhas))
```

Consumir de uma lista embaralhada (`pop`) garante que **nenhuma célula se
repete** — não pode haver um recurso em cima de uma armadilha.

**Por que fixo por seed?** (SPEC seção 6, item 3) Para o A\*, o Q-Learning e o GA
resolverem *o mesmo mapa*, tornando a comparação justa. Mesma `seed` → mesmo
mundo, sempre. Isso está testado em `test_mesma_seed_mesmo_layout`.

> **➜ Como mexer:**
> - Mudar o mapa inteiro: troque o `seed=`.
> - Grid maior/menor: `N=`. Cuidado: precisa de `2 + n_recursos + n_armadilhas`
>   células livres, senão `_gerar_layout` levanta `ValueError` de propósito.
> - Mapa **fixo escrito à mão** (útil para depurar): depois de criar o `env`,
>   sobrescreva `env.pos_inicial`, `env.pos_saida`, `env.recursos`,
>   `env.armadilhas` e `env.mascara_completa`, e chame `env.reset()`. É
>   exatamente o que os testes fazem em `montar_cenario()`.

---

## 5. O coração: o método `step()`

Todo passo executa, **nesta ordem**:

```
1. Calcula a célula-destino a partir da ação (dx, dy).
2. Se o destino está DENTRO do grid  -> move.
   Se está fora (borda)              -> permanece (movimento inválido).
3. passos += 1  e  energia -= 1      (SEMPRE, inclusive em movimento inválido).
4. Avalia a célula onde o agente parou -> recompensa base + possível término.
5. Se ainda não terminou: checa exaustão (energia<=0) e timeout (passos>=max).
```

O passo 2 implementa a regra "movimento inválido = fica no lugar e ainda gasta
energia" (SPEC 2.3): a energia cai no passo 3 **independentemente** de o
movimento ter sido válido.

### A avaliação da célula (`_avaliar_celula`)

É onde a tabela de recompensas da seção 2.4 vira código. A **ordem dos testes
importa** (o primeiro que casar vence):

```python
if pos in armadilhas:            return -50, terminado=True     # falha
if pos é recurso não coletado:   coleta; return +10             # continua
if pos == saida:
    if coletou tudo:             return +100, terminado=True    # sucesso
    else:                        return -1                      # não sai
return -1                                                       # passo comum
```

A armadilha é checada **primeiro** porque é terminal e domina tudo. A saída só
premia se o bitmask estiver completo — senão é tratada como piso comum e o
episódio continua.

### Exaustão e timeout SOMAM (a decisão desta entrega)

Depois da célula, se o episódio ainda não terminou:

```python
if energia <= 0:
    recompensa += -20      # SOMA, não sobrescreve
    terminado = True
elif passos >= max_passos:
    recompensa += -20
    terminado = True
```

Por isso uma coleta no último fôlego rende `+10 − 20 = −10`, e não `−20` seco. O
agente é **creditado por ter coletado**, mesmo perdendo por exaustão logo em
seguida — o que dá um sinal de aprendizado mais honesto para RL/GA. Note que
armadilha (−50) e sucesso (+100) já marcam `terminado=True` **antes** desse
bloco, então nunca recebem o −20 por cima.

> **➜ Como mexer nas recompensas:** todos os números vivem em `_avaliar_celula`
> e no bloco de exaustão/timeout de `step`. Quer punir mais o timeout? Troque o
> `-20`. Quer recompensar coleta com +20? Troque o `+10`. **Importante:** o SPEC
> (3.1) pede que o custo de passo do A\* "bata" com a recompensa do RL — se
> mexer no `-1` do passo comum, ajuste também o custo do A\* quando ele existir.

---

## 6. O `render()` — como o mapa é desenhado

`render()` monta uma string linha a linha. Para cada célula decide **um** símbolo
por prioridade:

```
A (agente)  >  X (armadilha)  >  S (saída)  >  R (recurso pendente)  >  . (vazio)
```

O agente vem primeiro porque, se ele está em cima de qualquer coisa, queremos ver
o **A**. Os recursos **pendentes** são recalculados na hora a partir do bitmask:

```python
recursos_pendentes = {
    celula for i, celula in enumerate(self.recursos)
    if not (self.coletados & (1 << i))
}
```

Por isso um `R` **some do mapa assim que é coletado** — ele deixa de estar no
conjunto de pendentes. Foi o que confirmamos visualmente (`00 → 01 → 11`).

O cabeçalho mostra `passos`, `energia`, o bitmask em binário e `terminado`.

> **➜ Como mexer:** o `render` devolve a string e só imprime se
> `mode == "human"`. Para capturar o mapa sem poluir a saída (ex.: em testes ou
> para salvar num log), chame `env.render(mode="silencioso")` e use o retorno.

---

## 7. A interface de agente (contrato que todos seguem)

Definida junto com o agente aleatório
([agentes/agente_aleatorio.py](agentes/agente_aleatorio.py)). Qualquer agente do
projeto precisa ter:

```python
class MeuAgente:
    def reset(self):                 # opcional: zera estado interno por episódio
        ...
    def escolher_acao(self, estado): # obrigatório: recebe estado, devolve 0..3
        ...
```

E a função genérica `rodar_episodio(env, agente)` roda um episódio inteiro e
devolve as **métricas do protocolo de avaliação** (SPEC seção 4):
`sucesso`, `recompensa`, `passos`, `energia_final`, `motivo`.

O agente aleatório é a implementação mínima desse contrato: `escolher_acao`
**ignora** o estado e devolve `rng.randrange(4)`. Por ignorar o estado, ele é o
**piso** — se um agente inteligente não superar isso, algo está errado.

> **➜ Como mexer / criar seu próprio agente:** copie a estrutura do
> `AgenteAleatorio`, e em `escolher_acao` use o `estado` para decidir. Como
> `rodar_episodio` é genérica, seu agente novo roda e é avaliado sem mudar mais
> nada. Foi assim que o teste `test_agente_de_sucesso_e_detectado` criou um
> agente "sempre-direita" em 3 linhas.

---

## 8. Como validar qualquer mudança que você fizer

O projeto tem uma rede de segurança. Depois de **qualquer** alteração:

```bash
python -m unittest discover -s tests -v
```

Se você mudou uma **regra** de propósito (ex.: recompensa de coleta de +10 para
+20), o teste correspondente vai **falhar** — isso é o esperado. Atualize o valor
esperado no teste (`tests/test_grid_world.py`) para refletir a nova regra. Um
teste que quebra ao mudar a regra é o teste fazendo o trabalho dele: te avisar de
todo efeito colateral.

Fluxo recomendado para mexer com segurança:
1. Mude o código.
2. Rode os testes → veja o que quebrou.
3. Se quebrou o que você **queria** mudar, ajuste o teste. Se quebrou algo
   **inesperado**, você achou um efeito colateral — investigue.
4. Verifique visualmente com `python -m ambiente.grid_world`.

---

## 9. Resumo de "onde fica cada coisa"

| Quero mudar... | Arquivo / lugar |
|---|---|
| Tamanho do grid, nº de recursos/armadilhas, energia | `GridWorldEnv.__init__` (parâmetros) |
| Como o mapa é sorteado | `_gerar_layout` |
| Ações disponíveis / movimento | dict `ACOES` + passo 2 de `step` |
| Valores de recompensa | `_avaliar_celula` + bloco exaustão/timeout de `step` |
| Condições de término | flags `terminado` em `_avaliar_celula` / `step` |
| Como o mapa é desenhado | `render` |
| Formato do estado devolvido | `_get_estado` |
| Comportamento do baseline | `AgenteAleatorio.escolher_acao` |
| Como um episódio é executado/medido | `rodar_episodio` |
| Algoritmo de busca / heurísticas | `AgenteBusca` em `agentes/agente_busca.py` |
| Cores, narração e velocidade da animação | `visualizacao.py` (`CORES`, `_descrever_evento`, `_montar_frame`) |

---

## 10. O agente de busca A\* (SPEC 3.1)

Arquivo: [agentes/agente_busca.py](agentes/agente_busca.py). É o primeiro agente
**inteligente** — mas atenção: ele **não aprende**. Ele **planeja**: dado o mapa,
calcula de antemão a sequência ótima de ações e depois só a executa. Se o mapa
mudar, ele replaneja do zero (por isso `reset()` chama `planejar()`).

### 10.1 O espaço de estados da busca

A grande sacada teórica (item 6 do vídeo): a busca **não** é sobre o grid físico
(100 células), é sobre o **grafo de estados compostos** `(posição, bitmask)`. Por
quê? Porque "estar em (3,4) sem nenhum recurso" é um problema diferente de "estar
em (3,4) já com 4 recursos" — o que falta fazer é diferente. Então o nó da busca
carrega os dois: onde estou **e** o que já coletei.

Tamanho: `100 posições × 32 bitmasks = 3200 nós` — pequeno, A\* resolve num piscar.
A **energia não entra** no nó de busca (SPEC 3.1): com energia inicial 300 ela
nunca limita o caminho ótimo, então planejamos sem ela.

### 10.2 Como o A\* funciona aqui (em 1 parágrafo)

A\* é Dijkstra + uma "bússola" (a heurística). Ele mantém uma fila de prioridade
ordenada por `f = g + h`, onde `g` = custo real já gasto (nº de passos) e
`h` = estimativa do que falta. Sempre expande o nó de menor `f`. Cada expansão
gera até 4 sucessores (as 4 ações); **armadilhas são tratadas como paredes** (não
viram sucessor), e mover para fora do grid é ignorado. Ao chegar ao objetivo
(`bitmask completo` **e** `posição == saída`), reconstrói a sequência de ações
seguindo o dicionário `came` de trás para frente.

### 10.3 As heurísticas e a admissibilidade (o coração teórico)

Uma heurística é **admissível** se **nunca superestima** o custo que falta. Isso
importa porque **A\* com heurística admissível é garantidamente ótimo**. Temos 4:

| Nome | `h(pos, restantes)` | Admissível? | Efeito |
|---|---|---|---|
| `nula` | `0` | Sim (trivial) | Vira Dijkstra: ótimo, mas expande tudo |
| `ingenua` | `dist(pos, saída)` | Sim (fraca) | Ignora recursos → pouca "bússola" |
| `admissivel` *(padrão)* | `max_r [dist(pos,r) + dist(r,saída)]` | **Sim** | Ótimo e informado |
| `guloso` | soma do tour vizinho-mais-próximo + saída | **Não** | Rápido, mas pode dar subótimo |

Por que `admissivel` é admissível? Para coletar um recurso `r` e depois chegar à
saída, qualquer caminho passa por `r` em algum momento e termina na saída, então
custa **pelo menos** `dist(pos,r) + dist(r,saída)`. Isso vale para **cada** `r`
pendente, então o **máximo** sobre eles ainda é um piso (lower bound) → admissível.

Por que `guloso` **não** é? Ela monta um tour concreto (vá ao recurso mais
próximo, depois ao próximo mais próximo...). O comprimento desse tour é **maior ou
igual** ao tour ótimo → pode **superestimar** → quebra a garantia de ótimo. O SPEC
sugeriu essa, mas aqui ela fica como opção *documentada* justamente para você
**ver a diferença** rodando `python -m agentes.agente_busca` (compare a coluna
"nós expandidos": `guloso` expande pouquíssimos, mas sem garantia).

> **➜ Como mexer:**
> - Trocar a heurística: `AgenteBusca(env, heuristica="ingenua")`.
> - Criar a sua: adicione um `if self.heuristica == "minha":` em `_h()` e o nome
>   em `HEURISTICAS`. Se quiser manter o A\* ótimo, garanta que ela seja um
>   **lower bound** do custo real.
> - Ver o efeito: o `__main__` do arquivo imprime nº de ações (custo) e nº de nós
>   expandidos por heurística — é a evidência de "informada = explora menos".

### 10.4 Por que ele se encaixa no mesmo `rodar_episodio`

Apesar de ser um planejador, o `AgenteBusca` respeita a **mesma interface** dos
outros: `reset()` planeja, e `escolher_acao(estado)` só devolve a próxima ação do
plano já calculado (`self.plano[self.idx]`). Assim ele roda no mesmíssimo
`rodar_episodio` do agente aleatório — e será medido pelo mesmo protocolo de
avaliação (seção 4) na comparação final.
