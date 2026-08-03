#!/usr/bin/env node
/**
 * CHANGELOG helper shared by the release pipelines of both packages.
 *
 * This is a monorepo: each package keeps its own CHANGELOG, so every command
 * takes the package it applies to (`python` → `sdk-python/CHANGELOG.md`,
 * `web` → `sdk-js-web/CHANGELOG.md`).
 *
 * Two commands:
 *
 *   node scripts/changelog.mjs notes <python|web> <version> [--allow-unreleased]
 *       Prints the body of the `## [<version>]` section (heading excluded) to
 *       stdout, for use as GitHub Release notes. With `--allow-unreleased` it
 *       falls back to the `## [Unreleased]` body when that version has no
 *       section yet (a release cut before the section was dated); without the
 *       flag it never does, so backfilling an old tag cannot staple the next
 *       cycle's notes onto it. When no section is found it prints a one-line
 *       pointer — a release is never blocked by missing notes.
 *
 *   node scripts/changelog.mjs close <python|web> <version> [<date>]
 *       Rewrites `## [Unreleased]` into `## [<version>] - <date>` in place and
 *       seeds a fresh empty `## [Unreleased]` above it, so the tag that goes out
 *       carries a dated section and the next cycle starts clean. No-ops when the
 *       version section already exists. `<date>` defaults to today (UTC).
 *
 * Exits non-zero only on genuine failure (bad usage, unreadable CHANGELOG),
 * never for "there was nothing to do".
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const UNRELEASED_HEADING = "## [Unreleased]";

/** Package directory per logical name, mirroring the release tag prefixes. */
const PACKAGE_DIRS = {
    python: "sdk-python",
    web: "sdk-js-web",
};

/** Escape a string for literal use inside a RegExp. */
function escapeRegExp(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Resolve the CHANGELOG path of one package.
 *
 * @param {string} pkg Logical package name (`python` or `web`).
 * @returns {string} Absolute path to that package's CHANGELOG.md.
 */
function changelogPath(pkg) {
    const dir = PACKAGE_DIRS[pkg];
    if (!dir) {
        console.error(`unknown package: ${pkg} (expected: ${Object.keys(PACKAGE_DIRS).join(" | ")})`);
        process.exit(1);
    }
    return join(ROOT, dir, "CHANGELOG.md");
}

/**
 * Slice the body of one `## [...]` section out of the CHANGELOG.
 *
 * Matches the heading loosely (`## [1.2.3]`, with or without a trailing date)
 * and stops at the next `## ` heading, so a section keeps all of its `###`
 * subsections.
 */
function sectionBody(text, version) {
    const heading = new RegExp(`^## \\[${escapeRegExp(version)}\\].*$`, "m");
    const start = text.search(heading);
    if (start === -1) return null;
    const after = text.slice(start);
    const nextHeading = after.slice(1).search(/^## /m);
    const body = nextHeading === -1 ? after : after.slice(0, nextHeading + 1);
    return body.replace(heading, "").trim();
}

/**
 * Body to use as Release notes for `version`.
 *
 * Only falls back to the `## [Unreleased]` body when `allowUnreleased` is set —
 * that fallback is for a release cut before the section was dated, and would be
 * plain wrong when backfilling an old tag (it would staple the next cycle's
 * notes onto a historical release).
 */
function notes(pkg, version, allowUnreleased) {
    const text = readFileSync(changelogPath(pkg), "utf8");
    const exact = sectionBody(text, version);
    if (exact) return exact;
    const unreleased = allowUnreleased ? sectionBody(text, "Unreleased") : null;
    if (unreleased) return unreleased;
    return `Sem entrada de CHANGELOG para ${version}. Veja o histórico completo em ${PACKAGE_DIRS[pkg]}/CHANGELOG.md.`;
}

/** Date the `## [Unreleased]` section as `version`, seeding a fresh one above it. */
function close(pkg, version, date) {
    const path = changelogPath(pkg);
    const text = readFileSync(path, "utf8");
    if (sectionBody(text, version) !== null) {
        return { changed: false, reason: `seção [${version}] já existe` };
    }
    if (!text.includes(UNRELEASED_HEADING)) {
        return { changed: false, reason: `${UNRELEASED_HEADING} não encontrado` };
    }
    const dated = `## [${version}] - ${date}`;
    writeFileSync(path, text.replace(UNRELEASED_HEADING, `${UNRELEASED_HEADING}\n\n${dated}`));
    return { changed: true, reason: `${UNRELEASED_HEADING} → ${dated}` };
}

const argv = process.argv.slice(2);
const allowUnreleased = argv.includes("--allow-unreleased");
const [command, pkg, version, date] = argv.filter((arg) => !arg.startsWith("--"));

if (!command || !pkg || !version) {
    console.error(
        "usage: changelog.mjs <notes|close> <python|web> <version> [date] [--allow-unreleased]",
    );
    process.exit(1);
}

if (command === "notes") {
    process.stdout.write(`${notes(pkg, version, allowUnreleased)}\n`);
} else if (command === "close") {
    const stamp = date ?? new Date().toISOString().slice(0, 10);
    const result = close(pkg, version, stamp);
    console.log(
        result.changed ? `✓ CHANGELOG: ${result.reason}` : `· CHANGELOG intacto (${result.reason})`,
    );
} else {
    console.error(`unknown command: ${command}`);
    process.exit(1);
}
