#!/usr/bin/env bash
# scripts/releases-check.sh — audita git tags × versões publicadas × GitHub Releases.
#
# Uso:
#   scripts/releases-check.sh
#
# Sai com status != 0 quando encontra dessincronia real, para o alvo do Makefile
# poder ser usado como gate. Tags listadas em `scripts/never-published.tsv` são
# ausência esperada e não derrubam o status.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEDGER="$ROOT/scripts/never-published.tsv"

PYPI_PKG="ort-vision-sdk"
NPM_PKG="@mauriciobenjamin700/ort-vision-sdk-web"

gh_releases=" $(gh release list --limit 200 --json tagName --jq '.[].tagName' 2>/dev/null | tr '\n' ' ') "
pypi_versions=" $(curl -sf "https://pypi.org/pypi/$PYPI_PKG/json" 2>/dev/null \
  | python3 -c "import json,sys; print(' '.join(json.load(sys.stdin)['releases']))" 2>/dev/null || true) "
npm_versions=" $(npm view "$NPM_PKG" versions --json 2>/dev/null | tr -d '[]", ' | tr '\n' ' ') "

ledger_tags=" $(grep -v '^[[:space:]]*#' "$LEDGER" 2>/dev/null | grep -v '^[[:space:]]*$' | cut -f1 | tr '\n' ' ') "

desync=0
declare -a notes=()

# Classifica uma tag e imprime a linha da tabela.
#
# Uma tag ausente do registry e presente no ledger é reportada como ausência
# esperada. O caso inverso também importa: tag no ledger que o registry serve
# significa que o ledger mentiu e precisa perder a linha — daí `LEDGER OBSOLETO`,
# que conta como dessincronia porque é informação errada sobre uma publicação.
check_tag() {
  local tag="$1" version="$2" registry_list="$3"
  local in_registry="FALTA" in_release="FALTA" status

  case "$registry_list" in *" $version "*) in_registry="ok" ;; esac
  case "$gh_releases" in *" $tag "*) in_release="ok" ;; esac

  if [[ "$in_registry" == "ok" && "$in_release" == "ok" ]]; then
    status="sincronizado"
    case "$ledger_tags" in
      *" $tag "*)
        status="LEDGER OBSOLETO"
        desync=1
        notes+=("$tag: está no registry mas consta em never-published.tsv — remova a linha de lá.")
        ;;
    esac
  else
    case "$ledger_tags" in
      *" $tag "*)
        in_registry="ausente"
        status="nunca publicada"
        notes+=("$tag: $(awk -F'\t' -v t="$tag" '$1 == t { print substr($0, index($0, "\t") + 1) }' "$LEDGER")")
        ;;
      *)
        status="DESSINCRONIZADO"
        desync=1
        ;;
    esac
  fi

  printf "%-16s %-10s %-10s %s\n" "$tag" "$in_registry" "$in_release" "$status"
}

printf "\n%-16s %-10s %-10s %s\n" "TAG" "REGISTRY" "RELEASE" "STATUS"

for tag in $(git -C "$ROOT" tag -l "v*.*.*" --sort=-v:refname); do
  check_tag "$tag" "${tag#v}" "$pypi_versions"
done

for tag in $(git -C "$ROOT" tag -l "web-v*.*.*" --sort=-v:refname); do
  check_tag "$tag" "${tag#web-v}" "$npm_versions"
done

if ((${#notes[@]})); then
  printf "\nNotas:\n"
  for note in "${notes[@]}"; do
    printf "  - %s\n" "$note"
  done
fi

if ((desync)); then
  printf "\n✗ Há dessincronia não registrada — veja as linhas acima.\n\n"
  exit 1
fi

printf "\n✓ Tags, registries e GitHub Releases em sincronia.\n\n"
