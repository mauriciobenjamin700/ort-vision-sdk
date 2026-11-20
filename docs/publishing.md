# Guia de publicação

Este projeto distribui dois pacotes a partir do mesmo monorepo, cada um isolado em seu próprio diretório:

| Pacote                | Registro | Diretório       | Tag de release             |
| --------------------- | -------- | --------------- | -------------------------- |
| `ort-vision-sdk`      | PyPI     | `sdk-python/`   | `v<MAJOR.MINOR.PATCH>`     |
| `@mauriciobenjamin700/ort-vision-sdk-web` | npm      | `sdk-js-web/`   | `web-v<MAJOR.MINOR.PATCH>` |

**Tudo é publicado a partir da sua máquina**, por targets do `Makefile`: os pacotes por [`scripts/publish.sh`](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/scripts/publish.sh), o site por `mkdocs gh-deploy`. Este repositório não usa GitHub Actions — os workflows em `.github/workflows/` estão desativados e ficam só como histórico. Empurrar uma tag **não publica nada**.

Há duas etapas: **credenciais** (uma vez por máquina) e **release** (toda vez que sai uma versão nova).

---

## Pré-requisitos

- `git`, [`uv`](https://docs.astral.sh/uv/), `node >= 20` com `npm`, e o [`gh`](https://cli.github.com) autenticado.
- Conta com 2FA em [pypi.org](https://pypi.org) e [npmjs.com](https://www.npmjs.com), com permissão de publicar os dois pacotes.

---

## 1. Credenciais

Nenhuma credencial entra no repositório. O `publish.sh` confere as duas **antes** de criar branch ou tag, e para com uma mensagem dizendo o que falta.

### 1.1. PyPI: token de API

1. Em <https://pypi.org/manage/account/token/>, crie um token com escopo **só do projeto** `ort-vision-sdk`.
2. Exporte no shell em que vai rodar o release:

    ```bash
    export UV_PUBLISH_TOKEN="pypi-..."
    ```

    O `uv publish` lê essa variável. Não grave o token em arquivo dentro do repositório.

### 1.2. npm: sessão

```bash
npm login
npm whoami   # tem de responder o seu usuário
```

!!! warning "Trusted Publishing e provenance ficaram para trás"
    OIDC (Trusted Publishing no PyPI, `--provenance` no npm) só existe dentro de um runner de CI. Publicando localmente, a autenticação é por token/sessão, e a página do pacote no npm não mostra mais a attestation de provenance.

---

## 2. Fluxo de release com `make`

O [Makefile](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/Makefile) delega para [scripts/release.sh](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/scripts/release.sh): cria uma **release branch**, faz bump de versão, valida, faz commit, cria a tag, **publica**, empurra branch e tag e abre um **PR** para a `main`.

### 2.1. Quick reference

```bash
make help                                # lista todos os alvos
make releases                            # histórico de tags por projeto
make last-python                         # última versão do sdk-python
make last-web                            # última versão do sdk-js-web

# Release completo (publica, empurra e abre PR):
make release PROJECT=python TAG=0.3.0
make release PROJECT=web    TAG=0.3.0

# Só publicar uma tag que já existe localmente:
make publish PROJECT=python TAG=0.3.0

# Site de docs:
make docs-build
make docs-deploy
```

> `TAG` é só o número da versão (ex.: `0.3.0`). O prefixo (`v` para Python, `web-v` para npm) é adicionado automaticamente.

### 2.2. Passo a passo de uma release

Exemplo: subir o sdk-python para `0.3.0`.

1. **Date o CHANGELOG** — em [sdk-python/CHANGELOG.md](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/sdk-python/CHANGELOG.md), mova o que está em `## [Unreleased]` para `## [0.3.0] - YYYY-MM-DD` e acrescente o link no fim do arquivo. Commit na `main` (o script exige working tree limpa):

    ```bash
    git add sdk-python/CHANGELOG.md
    git commit -m "docs: date the 0.3.0 changelog section"
    git push origin main
    ```

2. **Ensaie a publicação** — sem credencial e sem tag, nada sai da máquina:

    ```bash
    REHEARSAL=1 scripts/publish.sh python 0.3.0
    ```

    Roda as mesmas guardas, o build e o smoke-install que o publish de verdade, e para antes do upload. Como a versão ainda não foi bumpada, use a versão atual do pacote no ensaio.

3. **(Opcional) Dry-run do release:**

    ```bash
    make release PROJECT=python TAG=0.3.0 DRY_RUN=1
    ```

    Cria a branch `release/v0.3.0`, faz o bump, valida e cria a tag — **sem publicar, sem push, sem PR**. Para desfazer:

    ```bash
    git checkout main
    git branch -D release/v0.3.0
    git tag -d v0.3.0
    ```

4. **Rode o release:**

    ```bash
    make release PROJECT=python TAG=0.3.0
    ```

    Em ordem, o script:

    - confere o formato da `TAG` e que a tag não existe (local nem remoto);
    - avisa se o `CHANGELOG.md` não menciona `[0.3.0]`;
    - **confere a credencial** de publicação, antes de criar qualquer coisa;
    - cria a branch `release/v0.3.0`, faz o bump em `pyproject.toml` e `__init__.py`;
    - roda lint, typecheck, testes, build e `twine check` ([`scripts/validate.sh`](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/scripts/validate.sh));
    - commita `chore(python): release v0.3.0`, cria a tag `v0.3.0` e regenera o `RELEASES.md`;
    - **publica** com `scripts/publish.sh` (seção 2.4);
    - empurra branch e tag e abre o PR.

5. **Faça merge do PR** — ele leva para a `main` o bump de versão e o `RELEASES.md`. A versão já está publicada antes disso.

Para o npm é o mesmo fluxo, com `PROJECT=web`. O bump vai para `package.json`, `package-lock.json` e `src/index.ts`.

!!! danger "Publicar é irreversível"
    Nem o PyPI nem o npm deixam republicar uma versão. Se o publish falha **antes** do upload, branch e tag ficam só locais, e o script diz como descartar. Depois do upload, o caminho é yank/deprecate e um PATCH novo (seção 5).

### 2.3. Variáveis aceitas

| Variável | Descrição |
| --- | --- |
| `PROJECT=python\|web` | Qual SDK liberar (obrigatório em `make release` e `make publish`) |
| `TAG=0.3.0` | Versão sem prefixo, formato semver |
| `DRY_RUN=1` | Branch + commit + tag locais; pula publish, push e PR |
| `SKIP_VALIDATE=1` | Pula lint/typecheck/build (use só se acabou de validar) |
| `BASE_BRANCH=...` | Branch-alvo do PR (default `main`) |
| `REHEARSAL=1` | Só no `publish.sh`: guardas, build e smoke-install, sem upload |

### 2.4. O que o `publish.sh` faz

São as mesmas guardas dos antigos workflows, na mesma ordem, porque cada uma existe por um defeito que já aconteceu:

1. **Confere versão × tag.** `web-v0.4.0` só publica se `package.json` e `VERSION` disserem `0.4.0`; `v0.6.0` só publica se `pyproject.toml` **e** `__version__` disserem `0.6.0`. Divergência aborta antes do upload.
2. **Valida o artefato.** Build, `twine check` (Python) ou typecheck + testes (web), e smoke-install numa venv ou projeto limpo, importando as tasks, os envelopes de resultado e os helpers de grafo. O Python ainda confere que o wheel não saiu vazio.
3. **Publica** com `uv publish` ou `npm publish --access public`.
4. **Lê de volta do registry.** O PyPI é conferido pelo índice simples, por até 5 minutos; a API JSON fica atrasada logo depois do upload. O npm é conferido com `npm view --prefer-online`, versão e dist-tag `latest`, também por até 5 minutos.
5. **Cria ou atualiza o GitHub Release** da tag, com a seção do CHANGELOG como notas e o artefato anexado. É chamada de API do `gh`, não Actions.

!!! tip "Reconciliando o que ficou para trás"
    ```bash
    make releases-sync-dry   # lista os GitHub Releases que faltam
    make releases-sync       # cria os que faltam
    make releases-check      # tags × versões publicadas × Releases, lado a lado
    ```

    `releases-check` sai com status != 0 quando acha dessincronia, então serve de gate. Tags cuja publicação nunca aconteceu de propósito ficam em [`scripts/never-published.tsv`](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/scripts/never-published.tsv), uma linha por tag com o motivo, e o check as reporta como `nunca publicada` em vez de falhar.

### 2.5. Quando algo dá errado

- **`UV_PUBLISH_TOKEN vazio` / `npm sem sessão`** → seção 1. Nada foi criado.
- **`working tree sujo`** → commit ou stash antes.
- **`tag X já existe`** → `git tag -l`. Se foi engano e ela nunca foi publicada: `git tag -d <tag>`.
- **Validação falhou no meio** → o bump já está nos arquivos mas não no commit: `git checkout -- sdk-python/` (ou `sdk-js-web/`), conserte e rode de novo.
- **Publish falhou antes do upload** → branch e tag ficaram locais; conserte e rode `make publish PROJECT=... TAG=...`.
- **A leitura de volta falhou, mas a versão está no registry** → o publish deu certo e só a propagação atrasou. **Não** rode `make publish` de novo: o upload seria recusado por versão duplicada. Empurre a branch e a tag à mão, crie o GitHub Release com `make releases-sync` e abra o PR.

---

## 3. Site de docs

O site bilíngue (pt + en) é publicado no GitHub Pages pela branch `gh-pages`:

```bash
make docs-build    # mkdocs build --strict, sem publicar
make docs-deploy   # mkdocs gh-deploy --strict: builda e empurra a branch gh-pages
```

Os dois rodam num ambiente efêmero do `uv` montado de `docs/requirements.txt`, então não dependem do Python instalado na máquina. Mudança só de docs não ganha versão: commit `docs:` na `main` e `make docs-deploy`.

---

## 4. Versionamento

Os dois pacotes seguem [SemVer](https://semver.org). Mantenha-os em **lockstep** quando a mudança afeta os dois (ex.: novo tipo público), e independentes quando a mudança é só de um lado.

| Mudança                                    | Bump                |
| ------------------------------------------ | ------------------- |
| Bug fix sem mudança de API                 | PATCH (`0.1.x`)     |
| Nova função, novo parâmetro opcional       | MINOR (`0.x.0`)     |
| Remoção/renomeação/mudança de tipo público | MAJOR (`x.0.0`)     |
| Pré-1.0 (alpha)                            | qualquer break → MINOR é aceitável; documente no CHANGELOG. |

Locais que carregam a versão e precisam ficar em sincronia (o `release.sh` cuida disso):

- Python: [sdk-python/pyproject.toml](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/sdk-python/pyproject.toml) (`project.version`) e [sdk-python/src/ort_vision_sdk/\_\_init\_\_.py](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/sdk-python/src/ort_vision_sdk/__init__.py) (`__version__`).
- Web: [sdk-js-web/package.json](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/sdk-js-web/package.json) (`version`) e [sdk-js-web/src/index.ts](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/sdk-js-web/src/index.ts) (`VERSION`).

---

## 5. Rollback / yank

### PyPI

Não é possível **deletar** uma versão publicada. Se algo grave subir, faça **yank** em <https://pypi.org/manage/project/ort-vision-sdk/releases/>. Yank impede `pip install ort-vision-sdk` (sem versão) de pegar a versão problemática, mas quem pediu `==X.Y.Z` explicitamente ainda recebe — então **publique logo um PATCH corrigido**.

### npm

`npm` permite `unpublish` em até 72 h após a publicação:

```bash
npm unpublish @mauriciobenjamin700/ort-vision-sdk-web@0.2.0
```

Depois de 72 h, o caminho é `npm deprecate`:

```bash
npm deprecate @mauriciobenjamin700/ort-vision-sdk-web@0.2.0 "Critical bug — use 0.2.1+"
```

**Nunca** reaproveite um número de versão que já foi unpublished — o npm bloqueia.

---

## 6. Problemas comuns

**`twine check` reclama de README inválido**
A render do PyPI usa CommonMark estrito. Evite HTML cru e badges com URLs relativas.

**`uv publish` responde `403 Forbidden`**
O token não tem escopo para `ort-vision-sdk`, expirou ou foi revogado. Gere outro (seção 1.1).

**`npm publish` responde `E404` no `PUT`**
Sem sessão válida, o registry responde 404 para pacote escopado em vez de 401. Rode `npm whoami`; se falhar, `npm login`.

**Versão duplicada no PyPI/npm**
Os dois registros recusam reupload do mesmo número. Faça PATCH bump e publique de novo.

**O site não mudou depois do merge**
Merge não publica o site. Rode `make docs-deploy`.
