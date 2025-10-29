# Archived Scripts - Wave 1

**⚠️ DEPRECATED**: These scripts are from Wave 1 and have been replaced by the `./dev` CLI tool.

## Why Archived?

Wave 1 used multiple scattered scripts for different tasks. In Wave 2, we consolidated everything into a single, user-friendly CLI: `./dev`

## Migration Guide

| Old Script | New Command | Notes |
|-----------|-------------|-------|
| `setup_dev.sh` | `./dev setup` | One-command setup with Cloud SQL Proxy |
| `run_migrations.sh` | `./dev db-migrate` | Includes Cloud SQL Proxy checks |
| `start_server.sh` | `./dev server` | Includes environment loading |
| `test_weather_quick.sh` | `./dev test` | Run all tests with coverage |
| `validate_setup.sh` | `./dev db-check` | Verify Cloud SQL Proxy status |
| `validate_wave1.sh` | N/A | Wave 1 complete, no longer needed |
| `setup_gcp.sh` | N/A | One-time setup, managed by CI/CD now |
| `deploy.sh` | `git push` | Auto-deploys via GitHub Actions |
| Others | See `./dev help` | Check ./dev CLI for all commands |

## Key Changes

1. **Database**: Switched from Docker PostgreSQL (port 5432) to Cloud SQL Proxy (port 5433)
2. **Commands**: Consolidated 7+ scripts → 1 CLI tool
3. **User Experience**: Colored output, health checks, helpful error messages

## For Reference Only

These scripts are kept for historical reference but should NOT be used for new development.

**Use `./dev help` to see all available commands.**
