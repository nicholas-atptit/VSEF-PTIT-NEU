-- Table to store structured LLM-analyzed intelligence for news articles
CREATE TABLE IF NOT EXISTS news_intelligence (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    trend VARCHAR(20),     -- Bullish, Bearish, Neutral
    sentiment_score FLOAT, -- -1.0 to 1.0
    summary TEXT,          -- Concise summary
    full_report TEXT,      -- Detailed analysis/report
    article_ids TEXT[],    -- IDs/URLs of raw articles processed
    source_sites TEXT[],   -- List of sources (CafeF, VnExpress, etc.)
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast lookup by ticker
CREATE INDEX IF NOT EXISTS idx_news_intel_ticker ON news_intelligence (ticker, timestamp DESC);
