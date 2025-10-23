import pytest
from unittest.mock import patch, Mock
import pandas as pd
from gat_llm.tools.query_database import ToolQueryLLMDB, SampleOrder_LLM_DB


@pytest.fixture
def mock_duckdb_sql():
    with patch("duckdb.sql") as mock:
        mock.return_value.df = Mock(return_value=pd.DataFrame({"col1": range(10)}))
        yield mock


def test_unexpected_arg(unexpected_param_msg):
    db = SampleOrder_LLM_DB()
    tqd = ToolQueryLLMDB(db)
    result_gen = tqd("SELECT * FROM tblSales", unexpected_argument=None)
    for ans in result_gen:
        pass
    assert ans == f"{unexpected_param_msg}unexpected_argument"


def test_query_database_success(mock_duckdb_sql):
    db = SampleOrder_LLM_DB()
    tqd = ToolQueryLLMDB(db)
    result_gen = tqd("SELECT * FROM tblSales LIMIT 5")
    for result in result_gen:
        pass
    assert "SQL code executed correctly" in result
    assert "<query_results>" in result
    assert "</query_results>" in result


def test_query_database_too_many_records(mock_duckdb_sql):
    db = SampleOrder_LLM_DB()
    tqd = ToolQueryLLMDB(db, max_records=5)
    mock_duckdb_sql.df = pd.DataFrame({"col1": range(10)})
    result_gen = tqd("SELECT * FROM tblSales")
    for result in result_gen:
        pass
    assert "SQL code NOT executed. Too many records" in result
    assert "Number of records found: 10" in result


def test_query_database_error(mock_duckdb_sql):
    db = SampleOrder_LLM_DB()
    tqd = ToolQueryLLMDB(db)
    with patch("duckdb.sql", side_effect=Exception("SQL error")):
        result_gen = tqd("INVALID SQL")
        for result in result_gen:
            pass
    assert "SQL code NOT executed. Error description" in result
    assert "SQL error" in result


# Add more tests for different scenarios and edge cases
