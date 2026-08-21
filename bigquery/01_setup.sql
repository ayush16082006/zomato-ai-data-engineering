-- Zomato AI Data Engineering Project
-- BigQuery Initial Setup
-- Project: zomato-ai-data-engineering
-- Creates the datasets used by the data pipeline:
-- raw       -> Bronze / raw data
-- staging   -> Silver / cleaned data
-- marts     -> Gold / business-ready data
-- snapshots -> Historical/SCD2 data
-- ai        -> AI-enriched data


-- RAW / BRONZE
CREATE SCHEMA IF NOT EXISTS
`zomato-ai-data-engineering.raw`;


-- STAGING / SILVER
CREATE SCHEMA IF NOT EXISTS
`zomato-ai-data-engineering.staging`;


-- MARTS / GOLD
CREATE SCHEMA IF NOT EXISTS
`zomato-ai-data-engineering.marts`;


-- SNAPSHOTS / HISTORICAL DATA
CREATE SCHEMA IF NOT EXISTS
`zomato-ai-data-engineering.snapshots`;


-- AI-ENRICHED DATA
CREATE SCHEMA IF NOT EXISTS
`zomato-ai-data-engineering.ai`;