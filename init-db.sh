#!/bin/bash
set -e

echo "Creating pgvector extension..."

# Connect to the default database and enable pgvector
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS pgvector;
EOSQL

echo "pgvector extension enabled."