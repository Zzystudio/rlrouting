# AGENTS.md - rlrouting

This is a fresh repository with no build system, tests, or tooling configured yet.

## Project Structure

```
rlrouting/
├── src/        # empty - source code location (intended)
├── test/       # empty - test location (intended)
├── sim/        # empty - simulation code location (intended)
└── README.md   # GitLab template (not project-specific)
```

## Current State

- **No build system** - no package.json, pyproject.toml, Cargo.toml, go.mod, etc.
- **No test framework** - test/ directory exists but is empty
- **No linter/formatter/typechecker** configured
- **No CI/CD** configured
- **Language undecided** - empty src/, sim/, test/ directories suggest Python (sim/sim.py exists but empty) but no pyproject.toml/requirements.txt

## For Future Agents

When this project is initialized, update this file with:
- Language/framework and package manager
- Commands: `build`, `test`, `lint`, `typecheck`, `fmt`
- Test command to run a single test
- CI/CD pipeline details
- Any codegen, codegen, or special build steps