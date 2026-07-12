-- Runs once, automatically, when the `postgres` container initializes an
-- empty data directory (docker-entrypoint-initdb.d convention). Creates the
-- non-superuser role the api-gateway app connects as at runtime, so Postgres
-- row-level security (ENABLE + FORCE ROW LEVEL SECURITY, see
-- packages/anodyne-storage/src/anodyne_storage/db.py) is actually enforced:
-- the bootstrap `postgres` role is a SUPERUSER and superusers (and any role
-- with BYPASSRLS) bypass RLS unconditionally, even with FORCE.
--
-- Migrations (`make migrate` / Alembic) run as the `postgres` superuser and
-- own the tables; `anodyne_app` only gets DML privileges via the default
-- privileges grant below, applied automatically to tables Alembic creates
-- later in this same schema.
CREATE ROLE anodyne_app LOGIN PASSWORD 'anodyne_app' NOSUPERUSER;

GRANT USAGE ON SCHEMA public TO anodyne_app;

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO anodyne_app;
