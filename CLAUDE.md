# CLAUDE.md — ort-vision-sdk

Monorepo de **dois pacotes publicados** que espelham a mesma superfície: o
`ort-vision-sdk` no PyPI e o `@mauriciobenjamin700/ort-vision-sdk-web` no npm.
As regras globais de `~/.claude/CLAUDE.md` valem aqui; este arquivo registra o
que é específico deste repo — e o que já custou tempo descobrir.

## Layout

```text
ort-vision-sdk/
├── sdk-python/          # pacote PyPI — src/ort_vision_sdk/, tests/, CHANGELOG.md próprio
├── sdk-js-web/          # pacote npm — src/, tests via vitest, CHANGELOG.md próprio
├── docs/                # site MkDocs bilíngue COMPARTILHADO pelos dois (pt default + .en.md)
├── scripts/             # validate.sh, release.sh, gen_test_models.py, gen_parity_fixtures.py, bench.py
├── fixtures/            # fixtures de paridade Python × Web
├── bench/               # baseline de microbenchmark
└── Makefile             # release, validação, fixtures, benchmark
```

`sdk-python` usa `src/` layout apesar da regra global de layout flat para pacote
PyPI: aqui o `src/` separa o pacote dos seus `tests/` **dentro** do subdiretório
do monorepo, e `[tool.hatch.build.targets.wheel]` já aponta para ele. Não
"corrija" isso.

## A regra que este repo existe para não quebrar: paridade

**Os dois SDKs espelham a mesma superfície pública.** Uma mudança de
comportamento que entra só de um lado é como eles divergem — e a divergência é
silenciosa, porque cada suíte de testes só conhece o seu lado.

`sdk-python/tests/test_public_surface.py` é o guard: ele lê
`sdk-js-web/src/index.ts` e pareia cada export com um nome do
`ort_vision_sdk.__all__`. Exportar de um lado e não do outro **quebra a suíte**.
Assimetria legítima entra em `WEB_ONLY` ou `PYTHON_ONLY` **com o motivo escrito**
— é o registro da decisão, não uma lista de exceções para crescer sem pensar.

O guard cobre superfície, não comportamento. Os dois casos que motivaram esta
seção mostram a diferença:

- **Superfície (o guard pega hoje).** O root do Python exportava 45 nomes e o do
  web, os mesmos mais 34 — entre eles as oito exceções, então `except
  ModelLoadError` precisava de submódulo de um lado e nada do outro. A própria
  referência se contradizia: "tudo importável diretamente de `ort_vision_sdk`"
  três linhas acima de uma tabela apontando para `ort_vision_sdk.core`.
- **Comportamento (nenhum guard pega — só perguntar pega).** As tasks montavam
  o feed sempre em float32 e ninguém lia o `elem_type` do grafo, então modelo
  FP16 carregava e morria no primeiro `predict()`. O defeito era **idêntico dos
  dois lados**, e foi reportado só contra o web (#50) porque cada suíte
  alimentava fixtures que ela mesma exportou em float32. Um SDK "sem issue" não
  é um SDK sem o bug.

Ao mudar comportamento em `sdk-python/src/`, pergunte **sempre** se
`sdk-js-web/src/` precisa do mesmo — e vice-versa. Um hook `PreToolUse` avisa
quando um commit toca só um lado (`.claude/hooks/check-sdk-parity.sh`); ele
avisa, nunca bloqueia. Como ele roda **antes** do comando, `git add X && git
commit ...` num único comando é julgado pelo índice anterior ao próprio `add` —
faça o stage numa chamada e o commit na seguinte para o veredito valer. Quando a assimetria é legítima, **diga o porquê no corpo
do commit**:

- **Só Python:** `compose/` (fusão é build-time e precisa de `onnx`; não existe
  no web).
- **Só web:** `core/canvas.ts`, `warmup()`, tudo que depende de DOM/WebGPU.
- **Compartilhado:** `docs/` serve os dois; mudar prosa lá não conta como
  paridade de código.

## Validação

Rode o mesmo gate que o CI roda — não os comandos soltos:

```bash
make validate-python   # ruff check + ruff format --check + mypy + pytest + build + twine check
make validate-web      # tsc --noEmit + vitest + build + npm pack
```

Ambos passam nesta máquina. `scripts/validate.sh` é a implementação única que o
Makefile **e** o `release.sh` chamam, de propósito: duas cópias divergiriam
justo na do release.

Docs (bilíngue, obrigatório zero warning):

```bash
~/.pyenv/versions/3.13.4/bin/mkdocs build --strict
```

⚠️ O `mkdocs` do PATH é um shim do pyenv sem versão global — `mkdocs` puro
responde "command not found". Use o interpretador 3.13.4 (ou 3.12.11)
explicitamente.

Ao adicionar link cross-page, confira a âncora no HTML **buildado**
(`site/guia/<page>/index.html`): o MkDocs reporta âncora quebrada só como
`INFO`, então `--strict` passa com link morto.

## Release

O `release.sh` é **dono do bump de versão** — não edite `pyproject.toml` /
`__init__.py` / `package.json` à mão antes de cortar.

```bash
make release PROJECT=python TAG=0.9.0
make release PROJECT=web    TAG=0.8.0
DRY_RUN=1 make release ...     # tudo local, sem push
```

Antes de rodar, o `CHANGELOG.md` do pacote **já precisa** ter a seção
`## [TAG] - YYYY-MM-DD` (o script avisa se faltar) e a tabela de links no fim do
arquivo precisa da entrada nova.

⚠️ **O environment `pypi` não tem gate de aprovação.** Empurrar a tag publica
direto no PyPI — não existe passo manual entre uma coisa e outra.

Depois do push, valide o **artefato publicado**, não a árvore local: venv limpa
em diretório vazio, `uv pip install --no-cache --index-url https://pypi.org/simple "ort-vision-sdk==<versão>"`,
e exercite a superfície nova. A API JSON da PyPI mente sobre disponibilidade;
confirme pelo índice simples.

## Ferramentas nesta máquina

- **`jq` não está instalado.** Script de hook ou de CI que dependa dele falha em
  silêncio. `.claude/hooks/check-sdk-parity.sh` parseia o payload como texto por
  isso.
- **`gh issue view` e `gh issue close` quebram** com a deprecação de Projects
  classic (`repository.issue.projectCards`). `gh issue list` e `gh issue create`
  funcionam. Use REST para ler/fechar:
  `gh api /repos/mauriciobenjamin700/ort-vision-sdk/issues/<n> --jq '.title, "---", .body'`.
  O mesmo vale para `gh pr edit` — veja `~/.claude/rules/git-pr.md`.
- **Interpretador do pacote:** `sdk-python/.venv/bin/python`. O Makefile já cai
  nele automaticamente (`PY :=`), e usa `uv` quando o venv não existe.
- **Nunca rode `python` com o cwd dentro de `sdk-python/src/ort_vision_sdk/`.**
  O `types.py` do pacote sombreia o `types` da stdlib e o interpretador morre
  antes do seu script:

  ```text
  ImportError: cannot import name 'GenericAlias' from partially initialized
  module 'types' (most likely due to a circular import)
  ```

  Vale para qualquer script auxiliar, inclusive um `python3 - <<EOF` de uma
  linha. Rode da raiz do repo com caminho absoluto/relativo.
- **`Float16Array` não existe no Node do CI.** Chegou no V8 no Node 24; a matriz
  do CI roda 18, 20 e 22. Teste que constrói tensor half passa na sua máquina e
  **quebra no CI** — aconteceu no release do `web-v0.9.0`, que só não publicou
  errado porque o workflow roda os testes antes do `npm publish`. Teste de
  caminho half stuba o construtor; teste de recusa remove o global. Para
  conferir localmente antes de empurrar, rode a suíte com ele apagado:

  ```bash
  printf 'delete (globalThis as Record<string, unknown>)["Float16Array"];\n' > no-fp16.setup.ts
  printf 'import { defineConfig } from "vitest/config";\nexport default defineConfig({ test: { setupFiles: ["./no-fp16.setup.ts"] } });\n' > vitest.nofp16.config.ts
  npx vitest run --config vitest.nofp16.config.ts && rm no-fp16.setup.ts vitest.nofp16.config.ts
  ```

  (`vitest run --setupFiles` não existe no vitest 2 — só via config.)
- **O `npm view` mente logo depois de publicar**, do mesmo jeito que a API JSON
  da PyPI. `npm install <pkg>@<versão>` responde `notarget` por metadata em
  cache. Confirme pelo registry e instale com `--prefer-online`:

  ```bash
  curl -s https://registry.npmjs.org/@mauriciobenjamin700/ort-vision-sdk-web | grep -o '"latest":"[^"]*"'
  npm install --prefer-online "@mauriciobenjamin700/ort-vision-sdk-web@<versão>"
  ```
- **O gate linta só `src/`.** `ruff check src`, `ruff format --check src`,
  `mypy src` — `tests/` fica de fora de propósito (a suíte usa fakes com
  assinaturas que as regras `D`/`ANN` reprovariam). Rodar `ruff check tests` por
  conta própria produz centenas de achados que **não** são regressão, e
  `ruff format tests` reescreveria a suíte inteira num diff que ninguém pediu.

## Skills e agentes que valem aqui

| Situação | Use |
| --- | --- |
| Detalhar uma issue crua antes de atacar | `/detail-issue` |
| Agrupar mudanças em commits lógicos | `/commit` |
| Abrir/atualizar PR (template PT-BR global) | agente `pr-author` |
| Atacar o que acabou de ser implementado | `/contestar` |
| Revisar diff antes de cortar release | `/code-review` |

## Testes: o que a suíte já cobre e como

- `tests/test_compose.py` usa modelos **sintéticos**: o "detector" emite um head
  constante com caixas em coordenadas conhecidas e o "classificador" reduz cada
  recorte à sua média por canal. Isso é o que torna cada estágio da ponte
  observável de fora — a saída do classificador diz **quais pixels foram
  recortados**, coisa que nenhuma asserção de shape pegaria.
- `tests/test_parity.py` + `fixtures/` fixam a concordância Python × Web.
  Regerar com `make fixtures-parity` — e **revise o diff**, porque regerar
  silencia a divergência em vez de reportá-la.
- `tests/fixtures/models/*.onnx` vêm de `make fixtures-models` (precisa de
  `onnx`). Os valores esperados nos testes e2e são hard-coded de propósito: se
  uma fixture muda, o teste tem que ser atualizado deliberadamente.
- Modelo fundido: para provar o que o grafo realmente faz, exponha o tensor
  intermediário como saída extra (`model.graph.output.append(...)`) e rode. Foi
  assim que a divergência entre `boxes` e a ROI do `RoiAlign` ficou visível.
