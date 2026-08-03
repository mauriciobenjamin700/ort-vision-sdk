#!/usr/bin/env bash
# scripts/sync-github-releases.sh — reconcilia GitHub Releases com as git tags
# (e, por consequência, com as versões publicadas no PyPI/npm).
#
# Cada tag de release que ainda não tem Release ganha um, com as notas vindas da
# seção correspondente do CHANGELOG do pacote. Tags que já têm Release são
# puladas — o script é idempotente e nunca reescreve um Release existente.
#
# Este é um monorepo: `v*.*.*` é o sdk-python (PyPI) e `web-v*.*.*` é o
# sdk-js-web (npm). Rode sem argumento para reconciliar os dois.
#
# Uso:
#   scripts/sync-github-releases.sh                  # cria os Releases faltantes (ambos)
#   scripts/sync-github-releases.sh python           # só o sdk-python
#   scripts/sync-github-releases.sh web              # só o sdk-js-web
#   DRY_RUN=1 scripts/sync-github-releases.sh        # só lista o que faria
#
# Requer `gh` autenticado com permissão de escrita no repositório.

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
ONLY="${1:-all}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: gh CLI não encontrado"
  exit 1
fi

created=0
skipped=0

# Reconcilia um pacote: varre suas tags e cria os Releases que faltam.
#
# $1 nome lógico do pacote (python|web)
sync_package() {
  local pkg="$1" tag_glob prefix title_prefix registry_line version tag notes_file
  local -a flags

  case "$pkg" in
    python)
      tag_glob="v*.*.*"
      prefix="v"
      title_prefix="ort-vision-sdk (Python)"
      ;;
    web)
      tag_glob="web-v*.*.*"
      prefix="web-v"
      title_prefix="@mauriciobenjamin700/ort-vision-sdk-web"
      ;;
    *)
      echo "ERROR: pacote inválido '$pkg' — esperado python|web"
      exit 1
      ;;
  esac

  local -a tags=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && tags+=("$line")
  done < <(git tag -l "$tag_glob" --sort=v:refname)

  if [[ ${#tags[@]} -eq 0 ]]; then
    echo "· nenhuma tag ${tag_glob} encontrada"
    return 0
  fi

  for tag in "${tags[@]}"; do
    version="${tag#"$prefix"}"

    if gh release view "$tag" >/dev/null 2>&1; then
      skipped=$((skipped + 1))
      continue
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "→ criaria Release $tag (${title_prefix} ${version})"
      created=$((created + 1))
      continue
    fi

    notes_file="$(mktemp)"
    node scripts/changelog.mjs notes "$pkg" "$version" > "$notes_file"
    if [[ "$pkg" == "python" ]]; then
      registry_line="🐍 PyPI: [\`ort-vision-sdk==${version}\`](https://pypi.org/project/ort-vision-sdk/${version}/)"
    else
      registry_line="📦 npm: [\`@mauriciobenjamin700/ort-vision-sdk-web@${version}\`](https://www.npmjs.com/package/@mauriciobenjamin700/ort-vision-sdk-web/v/${version})"
    fi
    {
      echo ""
      echo "---"
      echo ""
      echo "$registry_line"
    } >> "$notes_file"

    flags=(--title "${title_prefix} ${version}" --notes-file "$notes_file" --verify-tag)
    # O badge "Latest" do GitHub é único no repositório, e num monorepo de dois
    # pacotes ele seguiria o que foi criado por último. Fica com o sdk-python.
    [[ "$pkg" == "web" ]] && flags+=(--latest=false)
    case "$version" in
      *[a-zA-Z]*) flags+=(--prerelease) ;;
    esac

    gh release create "$tag" "${flags[@]}" >/dev/null
    echo "✓ Release criado: $tag"
    created=$((created + 1))
    rm -f "$notes_file"
  done
}

case "$ONLY" in
  all) sync_package python; sync_package web ;;
  python|web) sync_package "$ONLY" ;;
  *) echo "ERROR: argumento inválido '$ONLY' — esperado python|web (ou nenhum)"; exit 1 ;;
esac

if [[ "$DRY_RUN" == "1" ]]; then
  printf "\n· dry-run: %d Release(s) faltando, %d já existentes\n" "$created" "$skipped"
else
  printf "\n✓ %d Release(s) criado(s), %d já existentes\n" "$created" "$skipped"
fi
