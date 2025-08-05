-- Initialize pgvector extension
CREATE EXTENSION IF NOT EXISTS pgvector;

-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS public;

-- Set search path
SET search_path TO public;

-- Create enum type for jurisdictions
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'jurisd') THEN
        CREATE TYPE jurisd AS ENUM ('BE', 'ES', 'DE');
    END IF;
END
$$;

-- Create documents table for storing tax documents
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR PRIMARY KEY,                      -- Format: "BE:20250804:AR-123"
    jurisdiction jurisd NOT NULL,                -- Country code (BE, ES, DE)
    source_system VARCHAR NOT NULL,              -- Source system identifier
    document_type VARCHAR NOT NULL,              -- Type of document
    title TEXT NOT NULL,                         -- Document title
    summary TEXT,                                -- Optional document summary
    issue_date DATE NOT NULL,                    -- Date when document was issued
    effective_date DATE,                         -- Optional date when document becomes effective
    language_orig VARCHAR(2) NOT NULL,           -- Original language of document (ISO 639-1)
    blob_url TEXT NOT NULL,                      -- URL to the document blob in storage
    checksum VARCHAR(64) UNIQUE NOT NULL,        -- Document checksum for integrity
    vector vector(1536),                         -- Vector embedding for semantic search
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Record creation timestamp
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS ix_jurisdiction_date ON documents(jurisdiction, issue_date);
CREATE INDEX IF NOT EXISTS ix_vector ON documents USING ivfflat (vector);

-- Create a function to list documents by jurisdiction
CREATE OR REPLACE FUNCTION list_documents_by_jurisdiction(jurisd_code jurisd)
RETURNS TABLE (
    id VARCHAR,
    title TEXT,
    issue_date DATE
) AS $$
BEGIN
    RETURN QUERY
    SELECT d.id, d.title, d.issue_date
    FROM documents d
    WHERE d.jurisdiction = jurisd_code
    ORDER BY d.issue_date DESC;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions to the taxdb user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO taxdb;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO taxdb;