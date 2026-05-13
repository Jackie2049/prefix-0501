# Version Control Policy

## Current source of truth

From 2026-05-13 onward, version management for this work uses the top-level
`prefix-0501` repository as the source of truth.

This means `survey/`, `dependency/`, and `refactor/prefix-sharing/` are tracked
together at the `prefix-0501` project granularity. Development commits should be
made from the top-level repository unless there is an explicit reason to update
a standalone upstream repository.

## prefix-sharing standalone history

`refactor/prefix-sharing/` was previously developed as a standalone Git
repository. Its standalone history has not been discarded:

- Standalone repository: `https://github.com/Jackie2049/prefix-sharing.git`
- Last standalone commit before the top-level migration:
  `af21cd0 [feat] 支持按样本前缀复用关系`

The nested `.git` directory is intentionally not tracked by `prefix-0501`.
It may remain locally as a migration reference, but new project-level changes
should be committed and pushed through `prefix-0501`.

When `prefix-sharing` is later split out for open source release, use either the
preserved standalone repository history above or split the current
`refactor/prefix-sharing/` path from `prefix-0501`, depending on the desired
public history shape.
