# Makefile — automação de release para os pacotes do monorepo
#
# Uso rápido:
#   make help                                  # lista todos os alvos
#   make releases                              # mostra histórico (a partir das git tags)
#   make release PROJECT=python TAG=0.3.0      # bump + validate + commit + tag + push
#   make release PROJECT=web    TAG=0.3.0      # idem para o pacote npm
#
# Variáveis aceitas:
#   PROJECT          python | web
#   TAG              número de versão sem prefixo (ex.: 0.3.0)
#   DRY_RUN=1        executa todo o pipeline mas NÃO faz push (cria branch + commit + tag locais)
#   SKIP_VALIDATE=1  pula o passo de validação (lint/typecheck/build)
#   BASE_BRANCH=...  branch alvo do PR (default: main; útil para PRs empilhados)

SHELL        := /bin/bash
.SHELLFLAGS  := -eu -o pipefail -c
.DEFAULT_GOAL := help

PROJECT       ?=
TAG           ?=
DRY_RUN       ?= 0
SKIP_VALIDATE ?= 0
BASE_BRANCH   ?= main

PY_DIR         := sdk-python
WEB_DIR        := sdk-js-web
RELEASES_FILE  := RELEASES.md

# Interpretador para os helpers em `scripts/`. Prefere o venv do pacote, que é
# onde as dependências deles já estão instaladas, e cai no `python` do PATH
# quando esse venv não existe. Sem isso, numa máquina com pyenv sem versão
# global todo alvo abaixo morre com "python: command not found" — o mesmo motivo
# que quebrava a validação do release.
PY := $(shell if [ -x $(PY_DIR)/.venv/bin/python ]; then echo $(PY_DIR)/.venv/bin/python; else echo python; fi)

PY_VERSION_FILES  := $(PY_DIR)/pyproject.toml $(PY_DIR)/src/ort_vision_sdk/__init__.py
WEB_VERSION_FILES := $(WEB_DIR)/package.json $(WEB_DIR)/package-lock.json $(WEB_DIR)/src/index.ts

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

.PHONY: help
help: ## Mostra esta ajuda
	@printf "Uso: make <alvo> [PROJECT=python|web] [TAG=0.3.0] [DRY_RUN=1] [SKIP_VALIDATE=1]\n\n"
	@printf "Alvos:\n"
	@awk 'BEGIN{FS=":.*?## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ---------------------------------------------------------------------------
# Histórico de releases (lê das git tags — fonte da verdade)
# ---------------------------------------------------------------------------

.PHONY: releases releases-python releases-web last-python last-web releases-md releases-check releases-sync releases-sync-dry

releases: ## Lista todas as tags de release agrupadas por projeto
	@printf "\n=== sdk-python (PyPI) ===\n"
	@git tag -l "v*.*.*" --sort=-v:refname | sed 's/^/  /' | grep . || echo "  (nenhuma tag ainda)"
	@printf "\n=== sdk-js-web (npm) ===\n"
	@git tag -l "web-v*.*.*" --sort=-v:refname | sed 's/^/  /' | grep . || echo "  (nenhuma tag ainda)"
	@printf "\n"

releases-python: ## Lista apenas as tags do sdk-python (mais recentes primeiro)
	@git tag -l "v*.*.*" --sort=-v:refname

releases-web: ## Lista apenas as tags do sdk-js-web (mais recentes primeiro)
	@git tag -l "web-v*.*.*" --sort=-v:refname

last-python: ## Mostra a última tag publicada do sdk-python
	@git tag -l "v*.*.*" --sort=-v:refname | head -n 1 | grep . || echo "(nenhuma)"

last-web: ## Mostra a última tag publicada do sdk-js-web
	@git tag -l "web-v*.*.*" --sort=-v:refname | head -n 1 | grep . || echo "(nenhuma)"

releases-check: ## Compara git tags × versões publicadas × GitHub Releases (falha se dessincronizado)
	@./scripts/releases-check.sh

releases-sync: ## Cria os GitHub Releases faltantes para as git tags existentes
	@./scripts/sync-github-releases.sh

releases-sync-dry: ## Mostra quais GitHub Releases faltam, sem criar nada
	@DRY_RUN=1 ./scripts/sync-github-releases.sh

releases-md: ## (Re)gera RELEASES.md a partir das git tags
	@{ \
	  printf "# Histórico de releases\n\n"; \
	  printf "_Gerado automaticamente por \`make releases-md\` a partir das git tags._\n\n"; \
	  printf "## sdk-python (PyPI)\n\n"; \
	  py_rows=$$(git for-each-ref --sort=-v:refname --format='| %(refname:short) | %(creatordate:short) | %(objectname:short) |' 'refs/tags/v*.*.*' 2>/dev/null || true); \
	  if [ -n "$$py_rows" ]; then \
	    printf "| Tag | Data | Commit |\n| --- | ---- | ------ |\n%s\n\n" "$$py_rows"; \
	  else \
	    printf "_Nenhuma release publicada ainda._\n\n"; \
	  fi; \
	  printf "## sdk-js-web (npm)\n\n"; \
	  web_rows=$$(git for-each-ref --sort=-v:refname --format='| %(refname:short) | %(creatordate:short) | %(objectname:short) |' 'refs/tags/web-v*.*.*' 2>/dev/null || true); \
	  if [ -n "$$web_rows" ]; then \
	    printf "| Tag | Data | Commit |\n| --- | ---- | ------ |\n%s\n" "$$web_rows"; \
	  else \
	    printf "_Nenhuma release publicada ainda._\n"; \
	  fi; \
	} > $(RELEASES_FILE)
	@echo "✓ $(RELEASES_FILE) atualizado"

# ---------------------------------------------------------------------------
# Bump de versão nos arquivos-fonte
# ---------------------------------------------------------------------------

.PHONY: bump-python bump-web

bump-python: _require-tag ## Atualiza versão do sdk-python (use TAG=0.3.0)
	@sed -i.bak -E 's/^version = "[^"]*"/version = "$(TAG)"/' $(PY_DIR)/pyproject.toml
	@sed -i.bak -E 's/^__version__: str = "[^"]*"/__version__: str = "$(TAG)"/' $(PY_DIR)/src/ort_vision_sdk/__init__.py
	@rm -f $(PY_DIR)/pyproject.toml.bak $(PY_DIR)/src/ort_vision_sdk/__init__.py.bak
	@echo "✓ sdk-python bumped → $(TAG)"

bump-web: _require-tag ## Atualiza versão do sdk-js-web (use TAG=0.3.0)
	@cd $(WEB_DIR) && npm version $(TAG) --no-git-tag-version --allow-same-version >/dev/null
	@sed -i.bak -E 's/^export const VERSION: string = "[^"]*";$$/export const VERSION: string = "$(TAG)";/' $(WEB_DIR)/src/index.ts
	@rm -f $(WEB_DIR)/src/index.ts.bak
	@echo "✓ sdk-js-web bumped → $(TAG)"

# ---------------------------------------------------------------------------
# Fixtures de teste e benchmarks
# ---------------------------------------------------------------------------
#
# O baseline de benchmark é uma referência LOCAL, não um gate de CI: runners
# compartilhados variam mais do que as regressões que vale detectar. Rode
# bench-python-check na mesma máquina que gravou o baseline, antes e depois da
# mudança.

.PHONY: fixtures-models fixtures-parity bench-python bench-python-save bench-python-check

BENCH_BASELINE := bench/baseline-python.json

fixtures-models: ## Regera os modelos ONNX sintéticos dos testes e2e (precisa de onnx)
	uv run --with onnx --with numpy python scripts/gen_test_models.py

fixtures-parity: ## Regera as fixtures de paridade Python×Web (revise o diff!)
	PYTHONPATH=$(PY_DIR)/src $(PY) scripts/gen_parity_fixtures.py

bench-python: ## Roda os microbenchmarks do sdk-python e imprime a tabela
	PYTHONPATH=$(PY_DIR)/src $(PY) scripts/bench.py

bench-python-save: ## Regrava o baseline de benchmark com os números desta máquina
	PYTHONPATH=$(PY_DIR)/src $(PY) scripts/bench.py --json $(BENCH_BASELINE)

bench-python-check: ## Compara os benchmarks com o baseline (rode na mesma máquina)
	PYTHONPATH=$(PY_DIR)/src $(PY) scripts/bench.py --compare $(BENCH_BASELINE)

# ---------------------------------------------------------------------------
# Validação local (mesmos checks que o CI roda)
# ---------------------------------------------------------------------------

.PHONY: validate-python validate-web

validate-python: ## Lint + typecheck + testes + build + twine check do sdk-python
	./scripts/validate.sh python

validate-web: ## Typecheck + testes + build + pack do sdk-js-web
	./scripts/validate.sh web

# ---------------------------------------------------------------------------
# Release pipeline
# ---------------------------------------------------------------------------

.PHONY: release release-python release-web

release: _require-project _require-tag ## Pipeline completo: make release PROJECT=python|web TAG=0.3.0
	@DRY_RUN=$(DRY_RUN) SKIP_VALIDATE=$(SKIP_VALIDATE) BASE_BRANCH=$(BASE_BRANCH) \
	  ./scripts/release.sh "$(PROJECT)" "$(TAG)"

release-python: _require-tag ## Release do sdk-python (use TAG=0.3.0)
	@DRY_RUN=$(DRY_RUN) SKIP_VALIDATE=$(SKIP_VALIDATE) BASE_BRANCH=$(BASE_BRANCH) \
	  ./scripts/release.sh python "$(TAG)"

release-web: _require-tag ## Release do sdk-js-web (use TAG=0.3.0)
	@DRY_RUN=$(DRY_RUN) SKIP_VALIDATE=$(SKIP_VALIDATE) BASE_BRANCH=$(BASE_BRANCH) \
	  ./scripts/release.sh web "$(TAG)"

# ---------------------------------------------------------------------------
# Guards (uso interno)
# ---------------------------------------------------------------------------

.PHONY: _require-tag _require-project

_require-tag:
	@test -n "$(TAG)" || { echo "ERROR: TAG é obrigatório (ex.: TAG=0.3.0)"; exit 1; }
	@echo "$(TAG)" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+([.-][a-zA-Z0-9]+)*$$' || \
	  { echo "ERROR: TAG inválido '$(TAG)' — esperado formato semver (ex.: 0.3.0, 1.0.0-rc1)"; exit 1; }

_require-project:
	@test -n "$(PROJECT)" || { echo "ERROR: PROJECT é obrigatório (python|web)"; exit 1; }
