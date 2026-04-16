-- 001: Initial schema — entities, triples, aliases, base indexes.
-- Matches the state before yoyo-migrations was introduced.
-- On existing databases, CREATE ... IF NOT EXISTS makes this a no-op.
-- depends:

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT DEFAULT 'unknown',
    properties TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS triples (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    confidence REAL DEFAULT 1.0,
    source_closet TEXT,
    source_file TEXT,
    extracted_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (subject) REFERENCES entities(id),
    FOREIGN KEY (object) REFERENCES entities(id)
);

CREATE TABLE IF NOT EXISTS entity_aliases (
    alias TEXT PRIMARY KEY,
    canonical_id TEXT NOT NULL,
    merged_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (canonical_id) REFERENCES entities(id)
);

CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject);
CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object);
CREATE INDEX IF NOT EXISTS idx_triples_predicate ON triples(predicate);
CREATE INDEX IF NOT EXISTS idx_triples_valid ON triples(valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_canonical ON entity_aliases(canonical_id);
