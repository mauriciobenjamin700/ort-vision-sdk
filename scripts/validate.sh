#!/usr/bin/env bash
# scripts/validate.sh — roda localmente os mesmos checks que o CI roda.
#
# Uso:
#   scripts/validate.sh python
#   scripts/validate.sh web
#
# Vive num script próprio em vez de inline no Makefile e no release.sh porque os
# dois precisam exatamente da mesma validação. Duas cópias divergiriam na
# primeira vez que alguém ajustasse uma delas — e a cópia que divergisse em
# silêncio seria justo a do release, que é a que não pode falhar.

set -euo pipefail

PROJECT="${1:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Roda os checks do pacote Python num ambiente que exista de fato na máquina,
# nesta ordem de preferência:
#
#   1. `sdk-python/.venv`, quando já existe — caminho instantâneo. O
#      `PYTHONPATH=src` não é decoração: esse venv pode carregar um editable
#      install apontando para OUTRO worktree, e sem forçar `src` na frente do
#      sys.path a suíte validaria o código de outra branch sem avisar.
#   2. `uv`, montando um ambiente efêmero a partir do `[dev]` do próprio pacote.
#      Não exige `python` no PATH nem `pip` dentro do venv — exatamente as duas
#      coisas que faltavam enquanto isso era `python -m pip install -e ".[dev]"`:
#      com pyenv sem versão global, `python` responde "command not found", e venv
#      criado pelo uv não traz `pip`. Nessa combinação a validação falhava
#      sempre, e no release.sh isso deixava uma branch órfã e nenhuma tag.
#   3. `pip` no interpretador atual — o caminho antigo, preservado para máquinas
#      que têm python+pip e não têm uv (o runner do CI, por exemplo).
validate_python() {
  local steps='ruff check src && ruff format --check src && mypy src && pytest -q && rm -rf dist && python -m build && twine check dist/*'

  cd "$ROOT/sdk-python"

  if [[ -x ".venv/bin/python" ]]; then
    echo "  (ambiente: sdk-python/.venv)"
    PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD/src" bash -c "$steps"
  elif command -v uv >/dev/null 2>&1; then
    echo "  (ambiente: efêmero, montado pelo uv)"
    uv run --no-project --with-editable ".[dev]" bash -c "$steps"
  else
    echo "  (ambiente: pip no interpretador atual)"
    python -m pip install --quiet -e ".[dev]"
    bash -c "$steps"
  fi
}

# O lado web não precisa de resolvedor: `npm ci` monta `node_modules` do lock, e
# todo binário usado sai de lá via `npm run`.
validate_web() {
  cd "$ROOT/sdk-js-web"
  npm ci
  npm run typecheck
  npm test
  npm run build
  npm pack --dry-run
}

case "$PROJECT" in
  python | py) validate_python ;;
  web | js | node) validate_web ;;
  *)
    echo "Uso: $0 <python|web>"
    exit 1
    ;;
esac
