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

Casos reais que passaram despercebidos exatamente assim:

- `OrtSession.providers` guardava a lista *pedida* nos dois SDKs. Corrigido no
  Python; o TypeScript ainda relata o pedido.
- `Classifier` assume normalização ImageNet nos dois, o que degrada em silêncio
  um classificador Ultralytics. Corrigido na fusão (Python-only); o
  `Classifier` de ambos os lados ainda tem o default antigo.

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
