# Guia de publicação

Este projeto distribui dois pacotes a partir do mesmo monorepo, cada um isolado em seu próprio diretório:

| Pacote                | Registro | Diretório       | Tag de release             |
| --------------------- | -------- | --------------- | -------------------------- |
| `ort-vision-sdk`      | PyPI     | `sdk-python/`   | `v<MAJOR.MINOR.PATCH>`     |
| `@ort-vision-sdk/web` | npm      | `sdk-js-web/`   | `web-v<MAJOR.MINOR.PATCH>` |

Os fluxos de release são automatizados em [.github/workflows/release-pypi.yml](../.github/workflows/release-pypi.yml) e [.github/workflows/release-npm.yml](../.github/workflows/release-npm.yml). Você publica empurrando uma tag — o GitHub Actions faz o resto.

Há duas etapas: **configuração inicial** (uma vez por pacote) e **release** (toda vez que sai uma versão nova).

---

## Pré-requisitos

- Repositório no GitHub: `https://github.com/mauriciobenjamin700/ort-vision-sdk` (ajuste se mudar de owner/repo).
- Conta com 2FA em [pypi.org](https://pypi.org) e [npmjs.com](https://www.npmjs.com).
- Local: `python >= 3.10`, `node >= 18`, `git`.

---

## 1. Configuração inicial — PyPI (Trusted Publishing)

Trusted Publishing usa OIDC: o GitHub Actions autentica direto no PyPI sem precisar de token armazenado.

### 1.1. Reserve o nome no TestPyPI (recomendado)

Antes do PyPI de verdade, valide tudo no [test.pypi.org](https://test.pypi.org):

1. Crie a conta em <https://test.pypi.org/account/register/>.
2. Em <https://test.pypi.org/manage/account/publishing/> → **Add a pending publisher**:
   - PyPI Project Name: `ort-vision-sdk`
   - Owner: `mauriciobenjamin700`
   - Repository name: `ort-vision-sdk`
   - Workflow name: `release-pypi.yml`
   - Environment name: `testpypi` (opcional; só se você criar esse environment no GitHub)

### 1.2. Trusted Publisher no PyPI

1. Crie a conta em <https://pypi.org/account/register/>.
2. Vá em <https://pypi.org/manage/account/publishing/> → **Add a pending publisher**:
   - PyPI Project Name: `ort-vision-sdk`
   - Owner: `mauriciobenjamin700`
   - Repository name: `ort-vision-sdk`
   - Workflow name: `release-pypi.yml`
   - Environment name: `pypi`

   > "Pending publisher" significa que o projeto ainda não existe no PyPI — ele será criado no primeiro upload.

### 1.3. Environment `pypi` no GitHub

1. No GitHub: **Settings → Environments → New environment**.
2. Nome: `pypi`.
3. (Opcional, mas recomendado) **Required reviewers** → adicione você mesmo: cada release exige aprovação manual antes de publicar.
4. (Opcional) **Deployment branches** → restrinja a `main` e tags `v*.*.*`.

Não precisa de secret nenhum aqui — OIDC cuida da autenticação.

---

## 2. Configuração inicial — npm

### 2.1. Decida o nome do pacote

O `package.json` atual usa `@ort-vision-sdk/web`. Esse escopo é uma **organização npm** que precisa existir e estar sob seu controle. Suas opções:

| Opção                                     | O que fazer                                                                                                                                                                                                |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Criar a org `ort-vision-sdk` no npm**   | <https://www.npmjs.com/org/create> → free tier. Mantém o nome atual.                                                                                                                                     |
| **Publicar no seu escopo pessoal**        | Renomeie em [sdk-js-web/package.json](../sdk-js-web/package.json) para `@mauriciobenjamin700/ort-vision-sdk-web` (seu username já é um escopo).                                                          |
| **Publicar sem escopo**                   | Renomeie para `ort-vision-sdk-web` (precisa estar disponível em <https://www.npmjs.com/package/ort-vision-sdk-web>). Nesse caso, remova `publishConfig.access` do `package.json` (irrelevante para nomes não-escopados). |

> Se renomear, atualize também o link `homepage` e os exemplos no [sdk-js-web/README.md](../sdk-js-web/README.md).

### 2.2. Gere o automation token

1. <https://www.npmjs.com/settings/SEU_USUARIO/tokens/granular-access-tokens/new>.
2. Tipo: **Granular access token** (preferido) ou **Classic → Automation**.
3. Permissions: **Read and write** no pacote (`@ort-vision-sdk/web` ou o nome que você escolheu).
4. Expiration: o que fizer sentido (ex.: 1 ano). Anote: o token só aparece **uma vez**.

### 2.3. Cadastre o secret no GitHub

1. **Settings → Secrets and variables → Actions → New repository secret**.
2. Nome: `NPM_TOKEN`.
3. Valor: o token gerado em 2.2.

### 2.4. (Opcional) Habilite provenance

O workflow já passa `--provenance`. Para isso funcionar:

- O pacote precisa ser **publicado a partir de um repositório público** com OIDC habilitado (já é o caso do GitHub Actions).
- O `repository.url` no `package.json` precisa apontar para o repo correto (já configurado).

Se preferir desabilitar, edite [.github/workflows/release-npm.yml](../.github/workflows/release-npm.yml) e remova `--provenance`.

---

## 3. Fluxo de release com `make` (recomendado)

O [Makefile](../Makefile) (que delega para [scripts/release.sh](../scripts/release.sh)) automatiza todo o fluxo: cria uma **release branch** dedicada, faz bump de versão, valida local, faz commit, cria a tag, faz push da branch + tag e abre um **PR** para a `main` via `gh`. A `main` nunca recebe push direto — toda mudança passa por revisão.

### 3.1. Quick reference

```bash
make help                                # lista todos os alvos disponíveis
make releases                            # mostra histórico de tags por projeto
make last-python                         # última versão publicada do sdk-python
make last-web                            # última versão publicada do sdk-js-web

# Release de verdade (cria release branch + tag + abre PR):
make release PROJECT=python TAG=0.3.0
make release PROJECT=web    TAG=0.3.0

# Atalhos equivalentes:
make release-python TAG=0.3.0
make release-web    TAG=0.3.0
```

> `TAG` é só o número da versão (ex.: `0.3.0`). O Makefile adiciona o prefixo certo (`v` para Python, `web-v` para npm) automaticamente.

### 3.2. Passo a passo de uma release

Exemplo: subir o sdk-python para `0.3.0`.

1. **Atualize o CHANGELOG** — em [sdk-python/CHANGELOG.md](../sdk-python/CHANGELOG.md), mova o que está em `## [Unreleased]` para `## [0.3.0] - YYYY-MM-DD`. Esse é o único passo manual.

2. **Faça commit do CHANGELOG** (o Makefile exige working tree limpa):

   ```bash
   git add sdk-python/CHANGELOG.md
   git commit -m "docs(python): changelog for 0.3.0"
   git push origin main   # ou pelo seu fluxo de PR habitual
   ```

3. **(Recomendado) Faça um dry-run primeiro:**

   ```bash
   make release PROJECT=python TAG=0.3.0 DRY_RUN=1
   ```

   Isso cria a branch `release/v0.3.0` localmente, atualiza a versão, valida e cria a tag — mas **não faz push nem abre PR**. Inspecione com `git log --oneline main..HEAD` e `git show v0.3.0`. Se algo estiver errado:

   ```bash
   git checkout main
   git branch -D release/v0.3.0
   git tag -d v0.3.0
   ```

4. **Rode o release de verdade:**

   ```bash
   make release PROJECT=python TAG=0.3.0
   ```

   O script vai, em ordem:
   - Validar que `TAG` está no formato semver e que a tag não existe (local nem remoto)
   - Avisar se o `CHANGELOG.md` não menciona `[0.3.0]`
   - Criar a branch `release/v0.3.0` a partir do HEAD atual
   - Atualizar a versão em `pyproject.toml` e em `src/ort_vision_sdk/__init__.py`
   - Rodar lint + typecheck + build + `twine check` (mesmos checks do CI)
   - Criar o commit `chore(python): release v0.3.0` e a tag `v0.3.0` na branch
   - Regenerar [RELEASES.md](../RELEASES.md) e fazer commit dele
   - `git push -u origin release/v0.3.0` + `git push origin v0.3.0`
   - Abrir um PR via `gh pr create` com o template padrão em PT-BR

5. **A tag já dispara o workflow** — acompanhe em **GitHub → Actions → Release to PyPI** e aprove o deploy no environment `pypi` quando solicitado. A publicação corre **independente** do merge do PR (a tag é a fonte da verdade).

6. **Faça merge do PR** quando estiver pronto — isso propaga para `main` o bump de versão e a entrada em `RELEASES.md`.

Para o npm é o mesmo fluxo, só trocando `PROJECT=python` por `PROJECT=web`. O Makefile adiciona o prefixo `web-v` na tag e atualiza `package.json`, `package-lock.json` e `src/index.ts`.

### 3.3. Variáveis aceitas

| Variável | Descrição |
| --- | --- |
| `PROJECT=python\|web` | Qual SDK liberar (obrigatório em `make release`) |
| `TAG=0.3.0` | Versão sem prefixo, formato semver |
| `DRY_RUN=1` | Faz tudo localmente (branch + commit + tag) mas pula push e PR |
| `SKIP_VALIDATE=1` | Pula lint/typecheck/build (use só se acabou de validar manualmente) |
| `BASE_BRANCH=...` | Branch-alvo do PR (default `main`). Útil para empilhar PRs (ex.: `BASE_BRANCH=release/v0.3.0` para o web encadear no python) |

### 3.4. Histórico de releases

A fonte da verdade é o `git tag`. Após cada release, o Makefile regenera [RELEASES.md](../RELEASES.md) com tag, data e SHA do commit por projeto. Para regenerar manualmente:

```bash
make releases-md
```

Para listar pelo terminal:

```bash
make releases          # ambos os projetos, mais recentes primeiro
make releases-python   # só Python
make releases-web      # só Web
make last-python       # só a última tag do Python
make last-web          # só a última tag do Web
```

### 3.5. Quando algo dá errado

- **`working tree sujo`** → faça commit ou stash antes de tentar de novo. O Makefile não atualiza versão por cima de mudanças pendentes (evita levar para o commit coisa que não devia).
- **`tag X já existe localmente`** → alguém (ou você) já criou essa tag. Veja com `git tag -l`. Se foi engano: `git tag -d <tag>` (e `git push origin :refs/tags/<tag>` caso já tenha sido pushed).
- **Validação falhou no meio do release** → a atualização de versão já foi aplicada nos arquivos mas não entrou no commit. Reverta com `git checkout -- sdk-python/` (ou `sdk-js-web/`), conserte o problema e rode de novo.
- **`CHANGELOG.md` não menciona [TAG]`** → o Makefile só avisa e espera você confirmar com ENTER. Se foi descuido, Ctrl+C, atualize o changelog, faça commit e rode de novo.

---

## 4. Fluxo manual — PyPI (referência)

> Use o `make release` da seção 3. Esta seção existe só como referência de "o que o Makefile está fazendo por baixo dos panos".

```bash
cd sdk-python

# 1. Atualize a versão em src/ort_vision_sdk/__init__.py e em pyproject.toml
#    (mantenha as duas em sincronia — ex.: 0.2.0)

# 2. Atualize o CHANGELOG.md (mova "Unreleased" para [0.2.0] - YYYY-MM-DD)

# 3. Valide localmente
python -m pip install -e ".[dev]"
ruff check src
ruff format --check src
mypy src
python -m build
twine check dist/*

# 4. Commit + tag (executados a partir da raiz do repositório)
cd ..
git add sdk-python/pyproject.toml sdk-python/src/ort_vision_sdk/__init__.py sdk-python/CHANGELOG.md
git commit -m "chore: release v0.2.0"
git tag v0.2.0
git push origin main
git push origin v0.2.0
```

A tag `v0.2.0` dispara o workflow `release-pypi.yml`:

1. Job **build** roda `python -m build` + `twine check` em `sdk-python/` e sobe os artefatos.
2. Job **publish** espera aprovação no environment `pypi` (se você habilitou required reviewers).
3. Aprove em **Actions → Release to PyPI → Review deployments**.
4. O `pypa/gh-action-pypi-publish` envia para o PyPI usando OIDC.

Verifique em <https://pypi.org/project/ort-vision-sdk/>. Para testar a instalação:

```bash
pip install ort-vision-sdk==0.2.0
python -c "from ort_vision_sdk import Classifier, Detector; print('OK')"
```

### 3.1. Testando antes — TestPyPI

Para fazer um ensaio sem queimar versão no PyPI real:

```bash
cd sdk-python
python -m build
python -m twine upload --repository testpypi dist/*
pip install -i https://test.pypi.org/simple/ ort-vision-sdk
```

Use credenciais do TestPyPI ou um token salvo em `~/.pypirc`:

```ini
[testpypi]
  username = __token__
  password = pypi-AgEN...   # token do test.pypi.org
```

---

## 5. Fluxo manual — npm (referência)

> Use o `make release` da seção 3. Esta seção existe só como referência de "o que o Makefile está fazendo por baixo dos panos".

```bash
cd sdk-js-web

# 1. Atualize a versão (npm version já cria o commit + tag)
#    Mas o workflow espera o prefixo `web-v`, então criamos a tag manualmente:
npm version 0.2.0 --no-git-tag-version

# 2. Atualize sdk-js-web/CHANGELOG.md (mova "Unreleased" para [0.2.0] - YYYY-MM-DD)

# 3. Valide localmente
npm ci
npm run typecheck
npm run build
npm pack --dry-run   # confira os arquivos que vão pro tarball

# 4. Commit + tag (a partir da raiz)
cd ..
git add sdk-js-web/package.json sdk-js-web/package-lock.json sdk-js-web/CHANGELOG.md
git commit -m "chore(web): release web-v0.2.0"
git tag web-v0.2.0
git push origin main
git push origin web-v0.2.0
```

A tag `web-v0.2.0` dispara o workflow `release-npm.yml`:

1. Checkout, `npm ci`, `npm run typecheck`, `npm run build` (em `sdk-js-web/`).
2. `npm publish --provenance --access public` autenticado pelo `NODE_AUTH_TOKEN` (`NPM_TOKEN`).

Verifique em <https://www.npmjs.com/package/@ort-vision-sdk/web>. Para testar:

```bash
mkdir /tmp/smoke && cd /tmp/smoke
npm init -y
npm install @ort-vision-sdk/web@0.2.0 onnxruntime-web
node -e "import('@ort-vision-sdk/web').then(m => console.log(Object.keys(m)))"
```

---

## 6. Versionamento

Os dois pacotes seguem [SemVer](https://semver.org). Mantenha-os em **lockstep** quando a mudança afeta os dois (ex.: novo tipo público), e independentes quando a mudança é só de um lado (ex.: ajuste só na preprocess do navegador).

| Mudança                                    | Bump                |
| ------------------------------------------ | ------------------- |
| Bug fix sem mudança de API                 | PATCH (`0.1.x`)     |
| Nova função, novo parâmetro opcional       | MINOR (`0.x.0`)     |
| Remoção/renomeação/mudança de tipo público | MAJOR (`x.0.0`)     |
| Pré-1.0 (alpha)                            | qualquer break → MINOR é aceitável; documente no CHANGELOG. |

Locais que carregam a versão e precisam ficar em sincronia:

- Python: [sdk-python/pyproject.toml](../sdk-python/pyproject.toml) (`project.version`) e [sdk-python/src/ort_vision_sdk/\_\_init\_\_.py](../sdk-python/src/ort_vision_sdk/__init__.py) (`__version__`).
- Web: [sdk-js-web/package.json](../sdk-js-web/package.json) (`version`) e [sdk-js-web/src/index.ts](../sdk-js-web/src/index.ts) (`VERSION`).

---

## 7. Checklist final antes de tagear

> Se você está usando `make release` (seção 3), todos esses itens são verificados/aplicados automaticamente — exceto o `CHANGELOG.md`, que continua sendo manual.

- [ ] Versão bumpada nos dois lugares do pacote (Python: `pyproject.toml` + `__init__.py`; Web: `package.json` + `index.ts`).
- [ ] `CHANGELOG.md` do pacote atualizado, com a data de hoje.
- [ ] `python -m build && twine check dist/*` passa em `sdk-python/`.
- [ ] `npm run typecheck && npm run build && npm pack --dry-run` passa em `sdk-js-web/`.
- [ ] CI verde no `main`.
- [ ] Tag empurrada com o prefixo certo (`v` para Python, `web-v` para npm).
- [ ] Workflow do GitHub Actions concluído com sucesso.
- [ ] `pip install` / `npm install` da versão recém-publicada funciona em ambiente limpo.

---

## 8. Rollback / yank

### PyPI

Não é possível **deletar** uma versão publicada (apenas removê-la imediatamente após o upload). Se algo grave subir, faça **yank**:

```bash
# Pelo site: https://pypi.org/manage/project/ort-vision-sdk/release/0.2.0/ → Yank
# Pela CLI:
twine yank ort-vision-sdk==0.2.0
```

Yank impede `pip install ort-vision-sdk` (sem versão) de pegar a versão problemática, mas quem pediu `==0.2.0` explicitamente ainda recebe — então **publique imediatamente uma 0.2.1 corrigida**.

### npm

`npm` permite `unpublish` em até 72 h após a publicação:

```bash
npm unpublish @ort-vision-sdk/web@0.2.0
```

Depois de 72 h, o caminho é `npm deprecate`:

```bash
npm deprecate @ort-vision-sdk/web@0.2.0 "Critical bug — use 0.2.1+"
```

E publicar uma versão patch corrigida. **Nunca** reaproveite um número de versão que já foi unpublished — o npm bloqueia.

---

## 9. Problemas comuns

**`twine check` reclama de README inválido**
A render do PyPI usa CommonMark estrito. Evite HTML cru e badges com URLs relativas — use URLs absolutas para imagens/links.

**`pypa/gh-action-pypi-publish` falha com `invalid-publisher`**
O Trusted Publisher não foi cadastrado, ou os campos (workflow, environment, repo) não batem com os do workflow. Confira em <https://pypi.org/manage/account/publishing/>.

**`npm publish` falha com `403 Forbidden — package name disputed`**
O nome (escopado ou não) já está em uso. Veja a seção 2.1 e renomeie.

**`npm publish` reclama de `provenance`**
Provenance só funciona se o repositório for público no GitHub e o workflow tiver `id-token: write`. Se for um repo privado, remova `--provenance` do workflow.

**Versão duplicada no PyPI/npm**
Os dois registros recusam reupload do mesmo número. Faça PATCH bump (`0.2.0` → `0.2.1`) e publique de novo.
