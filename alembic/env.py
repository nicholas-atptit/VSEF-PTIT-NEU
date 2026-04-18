"""Alembic environment configuration.

This repository is migration-first and SQL-first. Alembic should run against
the explicit migration history rather than treating ORM metadata as the
authoritative schema source.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from config.settings import get_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _resolved_database_url() -> str:
    """Resolve the sync database URL from application settings."""
    return get_settings().timescale_sync_url


config.set_main_option("sqlalchemy.url", _resolved_database_url().replace("%", "%%"))

# This codebase does not use ORM metadata as the canonical schema source.
target_metadata = None


def _configure_context(**kwargs: object) -> None:
    """Apply common Alembic context settings."""
    context.configure(
        target_metadata=target_metadata,
        compare_type=False,
        compare_server_default=False,
        **kwargs,
    )


def run_migrations_offline() -> None:
    """Run migrations in offline mode."""
    _configure_context(
        url=config.get_main_option("sqlalchemy.url"),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in online mode."""
    section = config.get_section(config.config_ini_section, {}) or {}
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        _configure_context(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
