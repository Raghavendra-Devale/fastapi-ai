-- Enable pgvector extension in the database
CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;

-- Create target AI schema
CREATE SCHEMA IF NOT EXISTS ai;

-- 1. Configuration/Model Versions Log
CREATE TABLE IF NOT EXISTS ai.ai_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider VARCHAR(50) NOT NULL,
    llm_model VARCHAR(100) NOT NULL,
    embedding_model VARCHAR(100) NOT NULL,
    prompt_version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Candidate Profiles Table
CREATE TABLE IF NOT EXISTS ai.candidate_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resume_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    headline VARCHAR(255),
    experience_years NUMERIC(4, 2),
    primary_role VARCHAR(100),
    profile_json JSONB,
    embedding VECTOR(384), -- Dimension of all-MiniLM-L6-v2 is 384
    embedding_model VARCHAR(100) NOT NULL,
    llm_model VARCHAR(100) NOT NULL,
    prompt_version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Unique index to guarantee 1-to-1 resume parsing mapping and query optimization
CREATE UNIQUE INDEX IF NOT EXISTS uq_candidate_profiles_resume_id ON ai.candidate_profiles (resume_id);
CREATE INDEX IF NOT EXISTS idx_candidate_profiles_user_id ON ai.candidate_profiles (user_id);

-- 3. Job Profiles Table
CREATE TABLE IF NOT EXISTS ai.job_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id BIGINT NOT NULL,
    provider VARCHAR(100),
    title VARCHAR(255),
    company VARCHAR(255),
    location VARCHAR(255),
    experience VARCHAR(100),
    profile_json JSONB,
    embedding VECTOR(384),
    embedding_model VARCHAR(100) NOT NULL,
    llm_model VARCHAR(100) NOT NULL,
    prompt_version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Unique index to guarantee 1-to-1 job parsing mapping and query optimization
CREATE UNIQUE INDEX IF NOT EXISTS uq_job_profiles_job_id ON ai.job_profiles (job_id);

-- 4. Recommendation Results Table
CREATE TABLE IF NOT EXISTS ai.recommendation_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_profile_id UUID NOT NULL REFERENCES ai.candidate_profiles(id) ON DELETE CASCADE,
    job_profile_id UUID NOT NULL REFERENCES ai.job_profiles(id) ON DELETE CASCADE,
    semantic_score NUMERIC(5, 4) NOT NULL,
    skill_score NUMERIC(5, 4) NOT NULL,
    experience_score NUMERIC(5, 4) NOT NULL,
    location_score NUMERIC(5, 4) NOT NULL,
    education_score NUMERIC(5, 4) NOT NULL,
    final_score NUMERIC(5, 4) NOT NULL,
    reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_recommendation_results_cp ON ai.recommendation_results (candidate_profile_id);
CREATE INDEX IF NOT EXISTS idx_recommendation_results_jp ON ai.recommendation_results (job_profile_id);

-- 5. Skill Gap Table
CREATE TABLE IF NOT EXISTS ai.skill_gap (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    recommendation_id UUID NOT NULL REFERENCES ai.recommendation_results(id) ON DELETE CASCADE,
    matched_required TEXT[],
    matched_preferred TEXT[],
    missing_required TEXT[],
    missing_preferred TEXT[],
    coverage NUMERIC(5, 4) NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_skill_gap_rec_id ON ai.skill_gap (recommendation_id);

-- 6. Processing Status / Background Job Tracker
CREATE TABLE IF NOT EXISTS ai.processing_status (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(50) NOT NULL, -- 'RESUME' or 'JOB'
    entity_id BIGINT NOT NULL,
    status VARCHAR(50) NOT NULL, -- 'PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'STALE'
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_processing_status_lookup ON ai.processing_status (entity_type, entity_id);
