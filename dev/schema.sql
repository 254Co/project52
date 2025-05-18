-- Create lookup tables for asset types, industries, currencies, sectors, countries, and exchanges
CREATE TABLE asset_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE industries (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL
);

CREATE TABLE sectors (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE currencies (
    code CHAR(3) PRIMARY KEY,
    name VARCHAR(50) NOT NULL
);

CREATE TABLE countries (
    code CHAR(2) PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

CREATE TABLE exchanges (
    id SERIAL PRIMARY KEY,
    code VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL
);

-- Create issuers table with references to countries and industries
CREATE TABLE issuers (
    issuer_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    lei VARCHAR(20) UNIQUE, -- Legal Entity Identifier
    country_code CHAR(2) REFERENCES countries(code),
    industry_id INT REFERENCES industries(id),
    metadata JSONB
);

-- Create securities table with reference to asset_types and self-reference for underlying securities
CREATE TABLE securities (
    security_id SERIAL PRIMARY KEY,
    figi VARCHAR(12) UNIQUE,
    isin VARCHAR(12),
    ticker VARCHAR(10),
    asset_type_id INT REFERENCES asset_types(id) NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    underlying_security_id INT REFERENCES securities(security_id),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ
);

-- Create identifiers table for additional security identifiers
CREATE TABLE identifiers (
    identifier_id SERIAL PRIMARY KEY,
    security_id INT REFERENCES securities(security_id),
    identifier_type VARCHAR(20), -- e.g., CUSIP, SEDOL
    identifier_value VARCHAR(50),
    UNIQUE (security_id, identifier_type)
);

-- Create reference_data table with references to exchanges, currencies, issuers, and sectors
CREATE TABLE reference_data (
    reference_id SERIAL PRIMARY KEY,
    security_id INT UNIQUE REFERENCES securities(security_id),
    exchange_id INT REFERENCES exchanges(id),
    currency_code CHAR(3) REFERENCES currencies(code),
    issuer_id INT REFERENCES issuers(issuer_id),
    sector_id INT REFERENCES sectors(id),
    listing_date DATE,
    metadata JSONB
);

-- Create corporate_actions table
CREATE TABLE corporate_actions (
    action_id SERIAL PRIMARY KEY,
    security_id INT REFERENCES securities(security_id),
    action_type VARCHAR(50), -- e.g., dividend, split
    effective_date DATE,
    details JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create audit_log table for tracking changes
CREATE TABLE audit_log (
    log_id SERIAL PRIMARY KEY,
    table_name VARCHAR(50),
    record_id INT,
    operation VARCHAR(20),
    changed_by VARCHAR(50),
    changed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    old_value JSONB,
    new_value JSONB
);

-- Trigger function for audit logging
CREATE OR REPLACE FUNCTION log_audit()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, record_id, operation, new_value)
        VALUES (TG_TABLE_NAME, NEW.security_id, 'INSERT', row_to_json(NEW));
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, record_id, operation, old_value, new_value)
        VALUES (TG_TABLE_NAME, NEW.security_id, 'UPDATE', row_to_json(OLD), row_to_json(NEW));
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, record_id, operation, old_value)
        VALUES (TG_TABLE_NAME, OLD.security_id, 'DELETE', row_to_json(OLD));
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Attach trigger to securities table
CREATE TRIGGER audit_securities
AFTER INSERT OR UPDATE OR DELETE ON securities
FOR EACH ROW EXECUTE FUNCTION log_audit();

-- Create indexes for performance and query optimization
CREATE INDEX idx_securities_figi ON securities(figi);
CREATE INDEX idx_securities_isin ON securities(isin);
CREATE INDEX idx_securities_ticker ON securities(ticker);
CREATE INDEX idx_securities_asset_type_id ON securities(asset_type_id);
CREATE INDEX idx_securities_underlying_security_id ON securities(underlying_security_id);
CREATE INDEX idx_issuers_country_code ON issuers(country_code);
CREATE INDEX idx_issuers_industry_id ON issuers(industry_id);
CREATE INDEX idx_reference_data_exchange_id ON reference_data(exchange_id);
CREATE INDEX idx_reference_data_currency_code ON reference_data(currency_code);
CREATE INDEX idx_reference_data_issuer_id ON reference_data(issuer_id);
CREATE INDEX idx_reference_data_sector_id ON reference_data(sector_id);
CREATE INDEX idx_corporate_actions_security_id ON corporate_actions(security_id);
CREATE INDEX idx_identifiers_value ON identifiers(identifier_value);