#!/usr/bin/env bash
# scripts/publish.sh — publica um pacote do monorepo a partir desta máquina.
#
# Uso:
#   scripts/publish.sh python <TAG>   # ort-vision-sdk no PyPI
#   scripts/publish.sh web <TAG>      # @mauriciobenjamin700/ort-vision-sdk-web no npm
#
# REHEARSAL=1 roda as guardas, o build e o smoke-install e para antes do
# upload: sem credencial, sem tag, nada sai da máquina.
#
# Credenciais (nunca entram no repositório):
#   python  UV_PUBLISH_TOKEN com um token de API do PyPI (escopo do projeto)
#   web     sessão do npm (`npm login`); `npm whoami` precisa responder
#
# Porta para cá os workflows `release-pypi.yml` e `release-npm.yml`, que estão
# desativados: fora da org 7rc este repositório não usa GitHub Actions. As
# guardas são as mesmas, na mesma ordem, porque cada uma existe por um defeito
# que já aconteceu:
#
#   1. versão do arquivo-fonte, do manifesto e da tag concordam — PyPI e npm
#      nunca aceitam republicar uma versão, então isso tem de falhar ANTES;
#   2. build, metadado e smoke-install do artefato numa venv/projeto limpo,
#      importando a superfície que importa (wheel vazio já foi publicado);
#   3. publish;
#   4. leitura de volta no registry, com a janela que a prática mostrou ser
#      necessária (o índice simples do PyPI e o `npm view --prefer-online`;
#      a API JSON do PyPI e o cache do npm mentem logo depois do publish);
#   5. GitHub Release da tag, com a seção do CHANGELOG e o artefato anexado.
#      É chamada de API do `gh`, não Actions.
#
# O que não dá para portar: Trusted Publishing (OIDC) e `--provenance` só
# existem dentro do runner. Aqui a autenticação é por token/sessão.
#
# A tag precisa existir localmente (o release.sh a cria); o passo 5 empurra só
# ela, se ainda não estiver no origin, porque o `gh release` exige a tag lá.

set -euo pipefail

PROJECT="${1:-}"
TAG="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "$PROJECT" || -z "$TAG" ]]; then
  echo "Uso: $0 <python|web> <TAG>"
  exit 1
fi

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

REHEARSAL="${REHEARSAL:-0}"

check_credentials() {
  [[ "$REHEARSAL" == "1" ]] && return 0
  case "$1" in
    python)
      [[ -n "${UV_PUBLISH_TOKEN:-}" ]] ||
        fail "UV_PUBLISH_TOKEN vazio — exporte um token de API do PyPI antes de publicar."
      command -v uv >/dev/null || fail "uv não encontrado no PATH."
      ;;
    web)
      npm whoami >/dev/null 2>&1 ||
        fail "npm sem sessão (npm whoami falhou) — rode 'npm login' antes de publicar."
      ;;
  esac
}

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  check_credentials "$PROJECT"
  echo "✓ credenciais de $PROJECT presentes"
  exit 0
fi

notes_file() {
  local project="$1" version="$2" footer="$3" out
  out="$(mktemp)"
  node "$ROOT/scripts/changelog.mjs" notes "$project" "$version" --allow-unreleased >"$out"
  printf '\n---\n\n%s\n' "$footer" >>"$out"
  echo "$out"
}

github_release() {
  local tag="$1" title="$2" notes="$3" prerelease="$4" latest="$5"
  shift 5
  if ! git ls-remote --tags origin "$tag" | grep -q "refs/tags/$tag\$"; then
    git push origin "$tag"
  fi
  local flags=(--title "$title" --notes-file "$notes")
  [[ "$prerelease" == "true" ]] && flags+=(--prerelease)
  if gh release view "$tag" >/dev/null 2>&1; then
    gh release edit "$tag" "${flags[@]}"
    gh release upload "$tag" "$@" --clobber
  else
    gh release create "$tag" "$@" "${flags[@]}" --latest="$latest"
  fi
  echo "✓ GitHub Release $tag"
}

publish_python() {
  local dir="$ROOT/sdk-python" version dunder git_tag="v$TAG" prerelease=false
  check_credentials python
  [[ "$REHEARSAL" == "1" ]] || git rev-parse "$git_tag" >/dev/null 2>&1 || fail "tag $git_tag não existe localmente."

  version=$(sed -nE 's/^version = "([^"]+)"/\1/p' "$dir/pyproject.toml" | head -1)
  dunder=$(sed -nE 's/^__version__: str = "([^"]+)"/\1/p' "$dir/src/ort_vision_sdk/__init__.py" | head -1)
  [[ "$dunder" == "$version" ]] || fail "__version__ $dunder não bate com pyproject $version."
  [[ "$version" == "$TAG" ]] || fail "tag $git_tag não bate com a versão $version."
  [[ "$version" == *[a-zA-Z]* ]] && prerelease=true

  echo "→ Build"
  rm -rf "$dir/dist"
  (cd "$dir" && uv build --quiet)
  uvx --quiet twine check "$dir"/dist/*

  echo "→ Smoke-install do wheel numa venv limpa"
  local smoke
  smoke="$(mktemp -d)"
  uv venv --quiet "$smoke/venv"
  uv pip install --quiet -p "$smoke/venv/bin/python" "$dir"/dist/*.whl
  (cd "$smoke" && "$smoke/venv/bin/python" -I -c "
import zipfile, glob, sys
import ort_vision_sdk as m
assert m.__version__ == '$version', m.__version__
assert m.Detector and m.Classifier and m.Segmenter, 'tasks missing'
assert m.Boxes and m.Probs and m.Masks, 'result envelopes missing'
assert m.spatial_input_size and m.resolve_input_size and m.model_names, 'graph helpers missing'
wheel = glob.glob(sys.argv[1])[0]
py = [n for n in zipfile.ZipFile(wheel).namelist() if n.startswith('ort_vision_sdk/') and n.endswith('.py')]
assert len(py) >= 10, f'wheel suspiciously empty ({len(py)} files)'
print('Smoke OK · version =', m.__version__, '·', len(py), 'arquivos .py no wheel')
" "$dir/dist/*.whl")
  rm -rf "$smoke"
  if [[ "$REHEARSAL" == "1" ]]; then
    echo "✓ REHEARSAL: ort-vision-sdk $version pronto para publicar (nada foi enviado)"
    return 0
  fi

  echo "→ Publish no PyPI"
  (cd "$dir" && uv publish dist/*)

  echo "→ Leitura de volta no índice simples do PyPI"
  local published=0
  for attempt in $(seq 1 30); do
    if curl -sf "https://pypi.org/simple/ort-vision-sdk/" | grep -q "ort_vision_sdk-${version}"; then
      published=1
      break
    fi
    echo "· índice simples ainda sem $version (tentativa $attempt); nova leitura em 10s"
    sleep 10
  done
  [[ "$published" == "1" ]] || fail "PyPI não serve ort-vision-sdk $version no índice simples."
  echo "✓ PyPI: ort-vision-sdk $version no índice simples"

  local notes
  notes=$(notes_file python "$version" \
    "🐍 PyPI: [\`ort-vision-sdk==${version}\`](https://pypi.org/project/ort-vision-sdk/${version}/) — publicado localmente por \`scripts/publish.sh\`.")
  github_release "$git_tag" "ort-vision-sdk (Python) ${version}" "$notes" "$prerelease" true "$dir"/dist/*
  rm -f "$notes"
}

# Roda a suíte web com `Float16Array` apagado do escopo global.
#
# O global só existe no Node 24. A matriz 20/22/24 do CI é que pegava teste de
# caminho half que passa aqui e quebra sem ele — foi assim no web-v0.9.0. Sem
# CI, a única máquina que roda a suíte antes do publish é esta, então a
# ausência do global é simulada. Os dois arquivos vivem dentro do pacote porque
# a config precisa resolver `vitest/config` de lá, e saem no fim mesmo se a
# suíte falhar.
without_float16() {
  local dir="$1" setup=".publish-nofp16.setup.ts" config=".publish-nofp16.config.mts"
  printf 'delete (globalThis as Record<string, unknown>)["Float16Array"];\n' >"$dir/$setup"
  printf 'import { defineConfig } from "vitest/config";\nexport default defineConfig({ test: { setupFiles: ["./%s"] } });\n' "$setup" >"$dir/$config"
  local status=0
  (cd "$dir" && npx vitest run --config "./$config" --silent) || status=$?
  rm -f "$dir/$setup" "$dir/$config"
  [[ "$status" == "0" ]] || fail "suíte web falhou sem Float16Array."
}

publish_web() {
  local dir="$ROOT/sdk-js-web" version src_version name git_tag="web-v$TAG" prerelease=false
  check_credentials web
  [[ "$REHEARSAL" == "1" ]] || git rev-parse "$git_tag" >/dev/null 2>&1 || fail "tag $git_tag não existe localmente."

  version=$(cd "$dir" && node -p "require('./package.json').version")
  src_version=$(sed -nE 's/^export const VERSION: string = "([^"]+)";$/\1/p' "$dir/src/index.ts" | head -1)
  name=$(cd "$dir" && node -p "require('./package.json').name")
  [[ "$src_version" == "$version" ]] || fail "VERSION $src_version não bate com package.json $version."
  [[ "$version" == "$TAG" ]] || fail "tag $git_tag não bate com a versão $version."
  [[ "$version" == *-* ]] && prerelease=true

  echo "→ Install, typecheck, testes e build"
  (cd "$dir" && npm ci --silent && npm run --silent typecheck && npm test --silent && npm run --silent build)

  echo "→ Testes de novo sem Float16Array (o que o Node 20/22 veria)"
  without_float16 "$dir"

  echo "→ Smoke-install do tarball num projeto limpo"
  local smoke tarball
  tarball=$(cd "$dir" && npm pack --silent)
  smoke="$(mktemp -d)"
  (
    cd "$smoke"
    npm init -y >/dev/null
    npm pkg set type=module >/dev/null
    npm install --silent "$dir/$tarball" "onnxruntime-web@^1.22"
    PKG_NAME="$name" EXPECTED="$version" node -e "import(process.env.PKG_NAME).then(m => {
      if (!m.Detector || !m.Segmenter || !m.Classifier) throw new Error('tasks missing');
      if (!m.Boxes || !m.Probs || !m.Masks) throw new Error('result envelopes missing');
      if (!m.spatialInputSize || !m.resolveInputSize) throw new Error('graph helpers missing');
      if (m.VERSION !== process.env.EXPECTED) throw new Error('VERSION ' + m.VERSION);
      console.log('Smoke OK · VERSION =', m.VERSION);
    })"
  )
  rm -rf "$smoke"
  if [[ "$REHEARSAL" == "1" ]]; then
    rm -f "$dir/$tarball"
    echo "✓ REHEARSAL: ${name}@${version} pronto para publicar (nada foi enviado)"
    return 0
  fi

  echo "→ Publish no npm"
  (cd "$dir" && npm publish "$tarball" --access public)

  echo "→ Leitura de volta no registry (até 5 min)"
  local published="" latest=""
  for attempt in $(seq 1 20); do
    published=$(npm view --prefer-online "${name}@${version}" version 2>/dev/null || true)
    [[ -n "$published" ]] && break
    echo "· registry ainda sem $version (tentativa $attempt); nova leitura em 15s"
    sleep 15
  done
  [[ "$published" == "$version" ]] || fail "npm não serve ${name}@${version} (veio '$published')."
  for attempt in $(seq 1 8); do
    latest=$(npm view --prefer-online "$name" dist-tags.latest 2>/dev/null || true)
    [[ "$latest" == "$version" ]] && break
    echo "· dist-tag latest ainda '$latest' (tentativa $attempt); nova leitura em 15s"
    sleep 15
  done
  [[ "$latest" == "$version" ]] || fail "dist-tag latest é '$latest', esperado '$version'."
  echo "✓ npm: ${name}@${version} publicado e marcado latest"

  local notes
  notes=$(notes_file web "$version" \
    "📦 npm: [\`${name}@${version}\`](https://www.npmjs.com/package/${name}/v/${version}) — publicado localmente por \`scripts/publish.sh\`.")
  github_release "$git_tag" "${name} ${version}" "$notes" "$prerelease" false "$dir/$tarball"
  rm -f "$notes" "$dir/$tarball"
}

case "$PROJECT" in
  python | py) publish_python ;;
  web | js | node) publish_web ;;
  *) fail "PROJECT '$PROJECT' inválido (use python ou web)" ;;
esac
