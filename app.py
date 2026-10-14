"""
Credit Risk Query Engine - Streamlit app
=========================================
Natural-language query engine for the Northbridge Bank commercial lending
portfolio (read-only PoC).  Converted from the notebook
"Learner Notebook - Project 3 - Credit Risk Query Engine (Solution)".

Pipeline (unchanged from the notebook):
    1. Intent classification  -> verified template (VQ1-VQ10) or generated SQL
    2. Query construction     -> library SQL, or LLM-generated read-only SQL
    3. Validation gate        -> 5 checks (read-only, schema, EXPLAIN dry run,
                                 LLM relevance, template integrity)
    4. Retry once (generated track only) if validation fails
    5. Escalate to a human analyst if still failing
    6. Execute on a read-only SQLite connection
    7. Business-focused narrative response + confidence score + audit log

Files expected next to this app.py:
    credit_risk_portfolio.db   (required)  SQLite database
    test_queries.csv           (optional)  ground-truth test cases

Configuration (Streamlit secrets or environment variables):
    OPENAI_API_KEY    (required)
    OPENAI_BASE_URL   (optional, for an OpenAI-compatible endpoint;
                       OPENAI_API_BASE is also accepted)
"""

import json
import os
import re
import sqlite3
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

# =============================================================================
# Configuration
# =============================================================================
APP_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("CREDIT_RISK_DB_PATH", APP_DIR / "credit_risk_portfolio.db"))
TEST_QUERIES_PATH = Path(os.environ.get("TEST_QUERIES_PATH", APP_DIR / "test_queries.csv"))

LLM_MODEL = "gpt-4o-mini"        # router, SQL generation, retry, narrative
EVALUATOR_MODEL = "gpt-4o"       # LLM relevance check in the validation gate


def get_config_value(*names):
    """Read a setting from Streamlit secrets first, then from environment variables."""
    for name in names:
        try:
            value = st.secrets[name]
        except Exception:
            value = None
        if not value:
            value = os.environ.get(name)
        if value:
            return str(value)
    return None


OPENAI_API_KEY = get_config_value("OPENAI_API_KEY")
OPENAI_BASE_URL = get_config_value("OPENAI_BASE_URL", "OPENAI_API_BASE")

# Same behaviour as the notebook: expose credentials as environment variables
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENAI_BASE_URL:
    os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL


def _client_kwargs():
    kwargs = {}
    if OPENAI_API_KEY:
        kwargs["api_key"] = OPENAI_API_KEY
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return kwargs


@st.cache_resource(show_spinner=False)
def get_llm():
    """llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)"""
    return ChatOpenAI(model=LLM_MODEL, temperature=0, **_client_kwargs())


@st.cache_resource(show_spinner=False)
def get_evaluator_llm():
    """evaluator_llm = ChatOpenAI(model='gpt-4o', temperature=0)"""
    return ChatOpenAI(model=EVALUATOR_MODEL, temperature=0, **_client_kwargs())


@contextmanager
def db_connection():
    """
    Read-only SQLite connection (URI mode with ?mode=ro prevents any write).
    A fresh connection is opened per pipeline run so concurrent Streamlit
    sessions never share one connection object.
    """
    uri = f"{DB_PATH.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def load_ground_truth(path_str, mtime):
    """ground_truth = pd.read_csv('test_queries.csv')  (mtime busts the cache on change)"""
    return pd.read_csv(path_str)


# =============================================================================
# Database schema provided to the LLM in the prompts
# =============================================================================
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""

# =============================================================================
# Verified Query Library (VQ1 - VQ10)
# Each query ID maps to a plain-English description (used for intent matching)
# and a fixed, pre-approved SQL statement that runs without modification.
# =============================================================================
verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': """ SELECT sm.sector_name,
    ROUND(SUM(lm.total_outstanding) / 1000000.0, 2) AS total_outstanding_mn,
    ROUND(SUM(CASE
                WHEN lm.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
                THEN lm.total_outstanding
                ELSE 0
              END) / 1000000.0, 2) AS npa_exposure_mn
FROM loan_master lm
JOIN sector_master sm
    ON lm.sector_code = sm.sector_code
GROUP BY sm.sector_name
ORDER BY total_outstanding_mn DESC"""
    },

    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': """SELECT loan_category,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mn,
    COUNT(*) AS loan_count
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_mn DESC"""
    },

    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': """ SELECT ifrs9_stage,
    COUNT(*) AS loan_count,
    ROUND(SUM(ead_amount) / 1000000.0, 2) AS total_ead_mn,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_mn
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage"""
    },

    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': """ SELECT sm.sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_provision_coverage_ratio
FROM provisioning p
JOIN loan_master lm
    ON p.loan_account_number = lm.loan_account_number
JOIN sector_master sm
    ON lm.sector_code = sm.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY sm.sector_name
ORDER BY avg_provision_coverage_ratio DESC"""
    },

    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': """ SELECT lm.borrower_name,
    sm.sector_name,
    ROUND(lm.total_outstanding / 1000000.0, 2) AS total_outstanding_mn,
    lm.asset_classification
FROM loan_master lm
JOIN sector_master sm
    ON lm.sector_code = sm.sector_code
ORDER BY lm.total_outstanding DESC
LIMIT 10"""
    },

    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': """SELECT group_name,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mn
FROM loan_master
WHERE group_name IS NOT NULL
GROUP BY group_name
ORDER BY SUM(total_outstanding) DESC
LIMIT 5"""
    },

    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': """SELECT lm.loan_account_number,
    lm.borrower_name,
    sm.sector_name,
    ROUND(lm.total_outstanding / 1000000.0, 2) AS total_outstanding_mn,
    lm.days_past_due,
    lm.asset_classification
FROM loan_master lm
JOIN sector_master sm
    ON lm.sector_code = sm.sector_code
WHERE lm.days_past_due > 0
ORDER BY lm.days_past_due DESC"""
    },

    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': """SELECT CASE
        WHEN days_past_due = 0               THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30  THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mn
FROM loan_master
GROUP BY dpd_bucket
ORDER BY MIN(days_past_due)"""
    },

    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': """SELECT borrower_id,
    previous_rating,
    internal_rating AS current_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC"""
    },

    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': """SELECT
    reporting_date,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_mn
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date"""
    }
}

# Display titles only (UI); the library above is the source of truth for routing/SQL
VQ_TITLES = {
    'VQ1': 'Sector-wise Outstanding and NPA Breakdown',
    'VQ2': 'Portfolio Outstanding by Loan Category',
    'VQ3': 'IFRS 9 Stage-wise ECL Summary',
    'VQ4': 'Provision Coverage Ratio by Sector',
    'VQ5': 'Top 10 Loan Exposures',
    'VQ6': 'Top 5 Business Group Exposures',
    'VQ7': 'All Overdue Loan Accounts',
    'VQ8': 'DPD Bucket Distribution',
    'VQ9': 'Latest Rating Downgrades',
    'VQ10': 'ECL Trend Across Reporting Quarters',
}


# =============================================================================
# Tool 1 - Intent Classification
# =============================================================================
def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated),'query_id' (template ID or None),
              and 'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
### ROLE
You are a query router for a credit risk analytics system. Your job is to decide whether a business user's question can be answered by one of the pre-approved query templates, or whether it needs fresh SQL generation.

### USER QUESTION
{user_question}

### VERIFIED QUERY TEMPLATES
{library_descriptions}

### INSTRUCTIONS
1. Return route "verified" when one template covers the metric, grouping, and time period the question is about. A template may return a broader result than the question asks for (for example, all sectors when the user asks about one sector); a later step narrows it down.
2. Return route "generated" when no template covers the question (for example, a different metric, a required filter such as a specific IFRS 9 stage or restructured loans, or a trend the templates do not provide).
3. For "generated", query_id must be null.

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = get_llm().invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# =============================================================================
# Tool 2 - Query Generation
# =============================================================================
def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""
### ROLE
You are a senior SQL developer specializing in credit risk analytics on a SQLite database.

### USER QUESTION
{user_question}

### DATABASE SCHEMA
{schema_context}

### INSTRUCTIONS
1. Write a single SQL query that answers the user question using only the provided schema.
2. The query must be read-only. Use SELECT (or WITH ... SELECT). Never use DROP, DELETE, UPDATE, INSERT, ALTER, or TRUNCATE.
3. Use only the tables and columns listed in the schema. Do not invent columns.
4. Resolve named entities using sector_name or sector_code where relevant (for example, 'Real Estate' to SEC_RE, 'Infrastructure' to SEC_INFRA).
5. Ensure the query is SQLite compatible.
6. In SQLite, never subtract DATE() or date columns directly (e.g. DATE(a)-DATE(b)) — it silently returns 0; always use julianday(a)-julianday(b) for day differences.
7. Alias every numeric column with a suffix that states its unit, so the result is self-describing. Use _usd for dollar amounts, _pct or _percent for percentages, _count for counts, _hours for hour values, and _days for day values. Avoid bare aliases like "value", "amount", or "total".

### OUTPUT
Return ONLY the SQL query, with no markdown code blocks, no comments, and no explanation.
"""

    sql = get_llm().invoke(generation_prompt).content.strip()
    # Strip markdown fences if present
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


# =============================================================================
# Tool 3 - Query Validation (five checks)
# =============================================================================
def _empty_result_probe(sql):
    """
    Wraps a query so that only its column structure is returned (zero rows).
    The notebook appended ' LIMIT 0' directly, which is a syntax error for
    templates that already end in LIMIT (VQ5, VQ6); a subquery wrapper works
    for every SELECT / WITH statement.
    """
    return f"SELECT * FROM ({sql.strip().rstrip(';').strip()}) LIMIT 0"


def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (int, 0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    # (as in the notebook, the unknown-identifier list is computed but advisory only;
    #  it does not fail validation because output aliases are legitimate identifiers)
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select', 'from', 'where', 'and', 'or', 'group', 'by', 'order', 'having', 'limit', 'join', 'on', 'as', 'case',
                    'when', 'then', 'else', 'end', 'sum', 'count', 'avg', 'min', 'max', 'round', 'desc', 'asc', 'left', 'right',
                    'inner', 'outer', 'distinct', 'null', 'is', 'not', 'in', 'like', 'with', 'union', 'all', 'between', 'coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    # prompt LLM to evaluate business relevance
    relevance_prompt = f"""
### ROLE
You are a senior data validator. Your job is to check whether a SQL query
correctly answers a business user's question about credit risk.

### CONTEXT
{track_context}

### INPUT
User question:
{user_question}

Candidate SQL:
{candidate_sql}

### INSTRUCTIONS
Assess whether the SQL genuinely answers what the user asked, considering:

1. Does it query the correct tables and columns?
2. Does it apply the right aggregations and groupings?
3. Does it handle the requested business definitions correctly?
4. Does it resolve named entities correctly?
5. Does it return the right shape of answer?
6. If this is a verified template, do not penalize it for returning
   a broader result set than the question's scope.

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

"""
    relevance_response = get_evaluator_llm().invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        try:
            expected_cols = [d[0] for d in cur.execute(_empty_result_probe(expected_sql)).description]
            actual_cols = [d[0] for d in cur.execute(_empty_result_probe(candidate_sql)).description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


# =============================================================================
# Tool 4 - Retry Generation
# =============================================================================
def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""
### ROLE
You are a senior SQL developer fixing a query that failed validation.

### INPUT
User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

### INSTRUCTIONS
1. Fix only the specific issue identified by the validation error.
2. Preserve the original intent of the query.
3. The revised SQL must be read-only SELECT (or WITH ... SELECT).
4. Use only tables and columns from the schema.
5. Ensure the query is SQLite compatible.

### OUTPUT
Return ONLY the corrected SQL, with no markdown code blocks, no comments, and no explanation.
"""

    revised_sql = get_llm().invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


# =============================================================================
# Tool 5 - Query Execution (deterministic, no LLM)
# =============================================================================
def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


# =============================================================================
# Tool 6 - Response Generation
# =============================================================================
def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""

### ROLE
You are a credit risk analyst writing a concise business response for a credit risk question.

### USER QUESTION
{user_question}

### QUERY RESULT
{dataframe.to_string()}

### INSTRUCTIONS
1. Answer the question directly, using exact figures from the query result. Columns ending in _mn are in USD millions.
2. The result may contain more rows than the question needs (for example, all sectors for a question about one sector); highlight only the relevant insights.
3. Keep the response concise and business-focused. Do not invent figures and do not mention SQL.
"""

    narrative = get_llm().invoke(response_prompt).content.strip()
    return narrative


# =============================================================================
# Pipeline Orchestration
# =============================================================================
def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, records the intermediate pipeline stages in 'trace'
                      (shown in the UI instead of being printed).

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, log and trace.
    '''

    log = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'execution_warnings': [],
        'confidence': None,
        'narrative': None
    }
    trace = []

    def note(message):
        if verbose:
            trace.append(message)

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    note(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    note(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    note(f"[2] Query Construction: {'loaded from library' if log['route'] == 'verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    note(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate['passed']:
        note(f"    Failed check: {gate.get('failed_check')}")
        note(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        note(f"[4] Retrying (generated track): {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        note(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            note(f"    Retry failed check: {gate.get('failed_check')}")
            note(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        note(f"[5] Escalated to human analyst: {gate['details']}")
        return {'log': log, 'dataframe': None, 'trace': trace, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)
    log['execution_warnings'] = exec_result['warnings']

    note(f"[6] Execute: {len(df)} rows returned")
    if exec_result['warnings']:
        note(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    note(f"[7] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, 'trace': trace, **log}


# =============================================================================
# Streamlit helpers
# =============================================================================
def md_safe(text):
    """Streamlit renders $...$ as LaTeX, which garbles dollar amounts - escape it."""
    return str(text).replace("$", "\\$")


def confidence_label(conf):
    if isinstance(conf, (int, float)):
        return f"{conf:.2f}"
    return str(conf) if conf is not None else "n/a"


def run_and_record(question):
    """Run the pipeline on a read-only connection and add the log to the audit trail."""
    with db_connection() as conn:
        result = run_pipeline(question, conn, verified_query_library, database_schema, verbose=True)
    st.session_state["history"].append(result["log"])
    return result


def render_result(result, key_prefix):
    """Show narrative, route, confidence, SQL used, raw data, trace and audit log."""
    log = result["log"]
    df = result.get("dataframe")

    if log["escalated"]:
        st.warning(f"🚨 {md_safe(log['narrative'])}")
    else:
        st.subheader("Answer")
        st.markdown(md_safe(log["narrative"]))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Route", (log["route"] or "n/a").title())
    c2.metric("Query ID", log["query_id"] or "—")
    c3.metric("Confidence", confidence_label(log["confidence"]))
    c4.metric("Rows returned", log["row_count"] if log["row_count"] is not None else "—")
    c5.metric("Retry used", "Yes" if log["retry_used"] else "No")
    if log.get("match_reason"):
        st.caption(f"Routing decision: {log['match_reason']}")

    for warning in log.get("execution_warnings", []):
        st.warning(f"Result check: {warning}")

    if log["escalated"]:
        st.subheader("Last candidate SQL (not executed)")
        st.code(log["candidate_sql"] or "", language="sql")
    else:
        st.subheader("SQL used")
        st.code(log["executed_sql"] or "", language="sql")

    if df is not None:
        st.subheader("Raw data returned")
        st.dataframe(df, hide_index=True)
        st.download_button(
            "Download result as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="query_result.csv",
            mime="text/csv",
            key=f"{key_prefix}_csv",
        )

    with st.expander("Pipeline trace"):
        st.code("\n".join(result.get("trace", [])), language=None)
    with st.expander("Audit log entry (JSON)"):
        audit_json = json.dumps(log, indent=2, default=str)
        st.code(audit_json, language="json")
        st.download_button(
            "Download audit entry",
            data=audit_json.encode("utf-8"),
            file_name="audit_entry.json",
            mime="application/json",
            key=f"{key_prefix}_audit",
        )


def ground_truth_checks(idx, result, gt_row):
    """
    Ground-truth comparisons carried over from the notebook for test cases 3-5
    (0-based position in test_queries.csv, as in the notebook's iloc[2], iloc[3], iloc[4]).
    """
    df = result.get("dataframe")
    if df is None or idx not in (2, 3, 4):
        return []
    checks = []
    try:
        if idx == 2:  # Restructured portfolio analysis
            restructured_count = len(df)
            impaired_count = int(df['is_impaired'].sum())
            impaired_pct = round(100 * impaired_count / restructured_count, 1)
            checks = [
                ("Route is generated", result['route'] == 'generated'),
                ("query_id is None", result['query_id'] is None),
                (f"35 restructured loans (actual {restructured_count})", restructured_count == 35),
                (f"24 impaired loans (actual {impaired_count})", impaired_count == 24),
                (f"Impaired % is 68.6 (actual {impaired_pct})", impaired_pct == 68.6),
            ]
        elif idx == 3:  # Average interest rate by sector
            rate_col = [c for c in df.columns if 'rate' in c.lower()][0]
            checks = [
                ("Route is generated", result['route'] == 'generated'),
                ("query_id is None", result['query_id'] is None),
                (f"10 sectors returned (actual {len(df)})", len(df) == 10),
                ("Sorted high to low", bool(df[rate_col].is_monotonic_decreasing)),
                ("No missing rates", bool(df[rate_col].notna().all())),
                ("Top sector is Pharma", 'Pharma' in str(df.iloc[0, 0])),
            ]
        elif idx == 4:  # Stage 3 trend over quarters
            cols = [c.lower() for c in df.columns]
            checks = [
                ("Route is generated", result['route'] == 'generated'),
                ("query_id is None", result['query_id'] is None),
                (f"4 quarters returned (actual {len(df)})", len(df) == 4),
                ("Has an EAD column", any('ead' in c for c in cols)),
                ("Has an ECL column", any('ecl' in c for c in cols)),
                ("Chronological order", bool(df.iloc[:, 0].is_monotonic_increasing)),
            ]
    except Exception as exc:  # the generated SQL may not have the expected columns
        checks.append((f"Ground-truth check could not be evaluated ({exc})", False))
    return checks


def build_evaluation(ground_truth, test_results):
    """Evaluation table + metrics, identical logic to the notebook's evaluation cell."""
    evaluation_rows = []
    for i, (_, gt) in enumerate(ground_truth.iterrows()):
        tr = test_results.get(i)
        if tr is None:
            continue
        evaluation_rows.append({
            'Test Case': gt['Test Case'],
            'Expected Route': gt['Expected Route'],
            'Actual Route': tr['route'],
            'Route Match': tr['route'] == gt['Expected Route'],
            'Expected Query ID': gt['Expected Query ID'],
            'Actual Query ID': tr['query_id'],
            'Query ID Match': (
                pd.isna(gt['Expected Query ID']) and pd.isna(tr['query_id'])
            ) or tr['query_id'] == gt['Expected Query ID'],
            'Confidence': tr['confidence'],
            'Rows Returned': tr['row_count'],
        })
    evaluation_df = pd.DataFrame(evaluation_rows)
    if evaluation_df.empty:
        return evaluation_df, None, None, None

    path_accuracy = evaluation_df['Route Match'].mean() * 100
    verified = evaluation_df['Expected Route'].astype(str).str.strip().str.lower() == 'verified'
    query_accuracy = (
        evaluation_df.loc[verified, 'Query ID Match'].mean() * 100 if verified.any() else float('nan')
    )
    average_confidence = pd.to_numeric(evaluation_df['Confidence'], errors='coerce').mean()
    return evaluation_df, path_accuracy, query_accuracy, average_confidence


# =============================================================================
# UI - tabs
# =============================================================================
def render_sidebar(api_ready):
    with st.sidebar:
        st.header("🏦 Credit Risk Query Engine")
        st.caption("Northbridge Bank · Commercial lending portfolio · Read-only PoC")

        st.markdown("**System status**")
        st.write(f"{'✅' if DB_PATH.exists() else '❌'} Database `{DB_PATH.name}` (read-only)")
        st.write(f"{'✅' if api_ready else '❌'} OpenAI API key")
        st.write(f"{'✅' if TEST_QUERIES_PATH.exists() else '⚠️'} Test cases `{TEST_QUERIES_PATH.name}`")

        st.markdown("**Models**")
        st.markdown(f"- Router / SQL / narrative: `{LLM_MODEL}`\n- Relevance validator: `{EVALUATOR_MODEL}`")

        st.markdown("**How it works**")
        st.markdown(
            "1. Classify the question → verified template or generated SQL\n"
            "2. Load the template, or generate read-only SQL\n"
            "3. Validate (read-only, schema, EXPLAIN, relevance, template integrity)\n"
            "4. Retry once if a generated query fails validation\n"
            "5. Escalate to a human analyst if it still fails\n"
            "6. Execute on a read-only connection\n"
            "7. Return narrative, SQL, raw data, confidence and an audit entry"
        )

        st.markdown("**Reference data**")
        st.markdown(
            "- Latest provisioning / rating date: 2025-09-30\n"
            "- NPA = Substandard, Doubtful, Loss"
        )


def render_ask_tab(ground_truth, api_ready):
    st.markdown(
        "Ask a routine portfolio question in plain English. Recurring questions are answered from "
        "pre-approved SQL templates; other questions get read-only generated SQL that must pass a "
        "validation gate. Every answer shows the SQL used, the raw data and a confidence score."
    )

    if ground_truth is not None and "User Query" in ground_truth.columns:
        examples = [str(q) for q in ground_truth["User Query"].dropna().tolist()]
    else:
        examples = [
            "Which sectors have the highest NPA exposure?",
            "Show me the days-past-due aging profile of the portfolio.",
            "What is the average interest rate by sector?",
        ]

    def use_example():
        st.session_state["question_input"] = st.session_state["example_choice"]

    ex_col, btn_col = st.columns([5, 1])
    ex_col.selectbox("Example questions", examples, key="example_choice")
    btn_col.write("")
    btn_col.write("")
    btn_col.button("Use example", on_click=use_example)

    question = st.text_area(
        "Your question",
        key="question_input",
        height=100,
        placeholder="e.g. What is the total outstanding and NPA exposure by sector?",
    )

    if st.button("Run query", type="primary", disabled=not api_ready):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Running the query engine pipeline…"):
                try:
                    st.session_state["last_result"] = run_and_record(question.strip())
                except Exception as exc:
                    st.session_state["last_result"] = None
                    st.error(f"The pipeline failed unexpectedly: {exc}")

    if st.session_state.get("last_result"):
        st.divider()
        render_result(st.session_state["last_result"], key_prefix="ask")


def render_library_tab():
    st.markdown(
        "The Verified Query Library holds ten pre-approved SQL templates. Each has a plain-English "
        "description used for intent matching and a fixed SQL statement that runs without modification."
    )
    for qid, entry in verified_query_library.items():
        with st.expander(f"{qid} — {VQ_TITLES.get(qid, '')}"):
            st.markdown(f"**Description:** {entry['description']}")
            st.code(entry["sql"].strip(), language="sql")
            if st.button("Preview result (runs the template directly, no LLM)", key=f"preview_{qid}"):
                try:
                    with db_connection() as conn:
                        res = execute_query(entry["sql"], conn)
                    for warning in res["warnings"]:
                        st.warning(warning)
                    st.dataframe(res["dataframe"], hide_index=True)
                except Exception as exc:
                    st.error(f"Could not execute the template: {exc}")


def render_evaluation_tab(ground_truth, api_ready):
    if ground_truth is None:
        st.info(
            f"`{TEST_QUERIES_PATH.name}` was not found next to app.py, so the evaluation harness is "
            "unavailable. Add it to the repository to enable this tab."
        )
        return

    st.markdown(
        "Ground-truth test cases used to check that the engine picks the right route and verified "
        "query, and returns the expected answer."
    )
    st.dataframe(ground_truth, hide_index=True)

    n_tests = len(ground_truth)
    labels = [
        f"{ground_truth['Test Case'].iloc[i]} — {str(ground_truth['User Query'].iloc[i])[:90]}"
        for i in range(n_tests)
    ]
    choice = st.selectbox("Select a test case", list(range(n_tests)), format_func=lambda i: labels[i])

    col_a, col_b = st.columns(2)
    run_one = col_a.button("Run selected test case", disabled=not api_ready, key="run_one")
    run_all = col_b.button("Run all test cases & evaluate", disabled=not api_ready, key="run_all")

    def run_test(i):
        question = str(ground_truth["User Query"].iloc[i])
        try:
            st.session_state["test_results"][i] = run_and_record(question)
            return True
        except Exception as exc:
            st.error(f"Test case {i + 1} failed unexpectedly: {exc}")
            return False

    if run_one:
        with st.spinner("Running test case…"):
            run_test(choice)

    if run_all:
        progress = st.progress(0.0, text="Running test cases…")
        for n in range(n_tests):
            run_test(n)
            progress.progress((n + 1) / n_tests, text=f"Completed {n + 1} of {n_tests} test cases")
        progress.empty()

    test_results = st.session_state["test_results"]

    if choice in test_results:
        st.divider()
        gt_row = ground_truth.iloc[choice]
        st.subheader(f"Result — {gt_row['Test Case']}")
        st.markdown(
            f"**Expected route:** {gt_row['Expected Route']}  \n"
            f"**Expected query ID:** {gt_row['Expected Query ID'] if pd.notna(gt_row['Expected Query ID']) else 'None'}  \n"
            f"**Expected answer:** {md_safe(gt_row['Expected Answer'])}"
        )
        render_result(test_results[choice], key_prefix=f"test{choice}")
        checks = ground_truth_checks(choice, test_results[choice], gt_row)
        if checks:
            st.subheader("Ground-truth checks")
            for label, ok in checks:
                st.write(f"{'✅' if ok else '❌'} {label}")

    evaluation_df, path_acc, query_acc, avg_conf = build_evaluation(ground_truth, test_results)
    if not evaluation_df.empty:
        st.divider()
        st.subheader("Evaluation against ground truth")
        st.caption(f"{len(evaluation_df)} of {n_tests} test cases run in this session")
        m1, m2, m3 = st.columns(3)
        m1.metric("Selected Path Accuracy", f"{path_acc:.1f}%")
        m2.metric("Selected Query Accuracy", "n/a" if pd.isna(query_acc) else f"{query_acc:.1f}%")
        m3.metric("Average Confidence Score", "n/a" if pd.isna(avg_conf) else f"{avg_conf:.2f}")
        st.dataframe(evaluation_df, hide_index=True)


def render_audit_tab():
    st.markdown(
        "Every question run in this session is logged here with its route, SQL, validation outcome "
        "and narrative. Download the log to keep a permanent record (the log is not stored on the server)."
    )
    history = st.session_state["history"]
    if not history:
        st.info("No queries have been run yet.")
        return

    summary_cols = [
        "timestamp_utc", "user_question", "route", "query_id", "confidence",
        "retry_used", "escalated", "row_count", "executed_sql",
    ]
    summary = pd.DataFrame(history).reindex(columns=summary_cols)
    st.dataframe(summary, hide_index=True)

    d1, d2 = st.columns(2)
    d1.download_button(
        "Download summary (CSV)",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="audit_summary.csv",
        mime="text/csv",
    )
    d2.download_button(
        "Download full audit log (JSON)",
        data=json.dumps(history, indent=2, default=str).encode("utf-8"),
        file_name="audit_log.json",
        mime="application/json",
    )


# =============================================================================
# Main
# =============================================================================
def main():
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("test_results", {})
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("question_input", "")

    api_ready = bool(OPENAI_API_KEY)
    render_sidebar(api_ready)

    st.title("🏦 Credit Risk Query Engine")
    st.caption("Plain-English portfolio questions · verified answers · transparent SQL · full audit trail")

    if not DB_PATH.exists():
        st.error(
            f"Database file not found: `{DB_PATH}`. Place `credit_risk_portfolio.db` next to app.py "
            "(or set the CREDIT_RISK_DB_PATH environment variable)."
        )
        st.stop()

    if not api_ready:
        st.warning(
            "OPENAI_API_KEY is not configured. Add it to Streamlit secrets (or as an environment "
            "variable) to run questions. The Verified Query Library tab still works without it."
        )

    ground_truth = None
    if TEST_QUERIES_PATH.exists():
        ground_truth = load_ground_truth(str(TEST_QUERIES_PATH), TEST_QUERIES_PATH.stat().st_mtime)

    tab_ask, tab_library, tab_eval, tab_audit = st.tabs(
        ["💬 Ask a question", "📚 Verified query library", "🧪 Test cases & evaluation", "🧾 Audit log"]
    )
    with tab_ask:
        render_ask_tab(ground_truth, api_ready)
    with tab_library:
        render_library_tab()
    with tab_eval:
        render_evaluation_tab(ground_truth, api_ready)
    with tab_audit:
        render_audit_tab()


if __name__ == "__main__":
    main()
