import pytest
from unittest.mock import patch, Mock
import pandas as pd
from gat_llm.tools.query_database import (
    ToolQueryLLMDB,
    SampleOrder_LLM_DB,
    validate_sql_query,
    SQLValidationError,
    BLOCKED_SQL_KEYWORDS,
)


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


# SQL Validation Tests


class TestSQLValidation:
    """Tests for SQL keyword blocking functionality."""

    def test_valid_select_query_passes(self):
        """Valid SELECT queries should not raise exceptions."""
        valid_queries = [
            "SELECT * FROM tblSales",
            "SELECT col1, col2 FROM tblSales WHERE col1 > 5",
            "SELECT COUNT(*) FROM tblSales GROUP BY category",
            "SELECT * FROM tblSales ORDER BY date DESC LIMIT 10",
            "SELECT a.col1, b.col2 FROM tbl1 a JOIN tbl2 b ON a.id = b.id",
        ]
        for query in valid_queries:
            validate_sql_query(query)  # Should not raise

    @pytest.mark.parametrize("keyword", BLOCKED_SQL_KEYWORDS)
    def test_blocked_keywords_rejected(self, keyword):
        """Each blocked keyword should raise SQLValidationError."""
        # Create a query using the blocked keyword
        query = f"{keyword} something"
        with pytest.raises(SQLValidationError) as exc_info:
            validate_sql_query(query)
        assert keyword in str(exc_info.value)

    def test_drop_table_blocked(self):
        """DROP TABLE should be blocked."""
        with pytest.raises(SQLValidationError) as exc_info:
            validate_sql_query("DROP TABLE users")
        assert "DROP" in str(exc_info.value)

    def test_delete_blocked(self):
        """DELETE statement should be blocked."""
        with pytest.raises(SQLValidationError) as exc_info:
            validate_sql_query("DELETE FROM users WHERE id = 1")
        assert "DELETE" in str(exc_info.value)

    def test_update_blocked(self):
        """UPDATE statement should be blocked."""
        with pytest.raises(SQLValidationError) as exc_info:
            validate_sql_query("UPDATE users SET name = 'test'")
        assert "UPDATE" in str(exc_info.value)

    def test_insert_blocked(self):
        """INSERT statement should be blocked."""
        with pytest.raises(SQLValidationError) as exc_info:
            validate_sql_query("INSERT INTO users VALUES (1, 'test')")
        assert "INSERT" in str(exc_info.value)

    def test_create_blocked(self):
        """CREATE statement should be blocked."""
        with pytest.raises(SQLValidationError) as exc_info:
            validate_sql_query("CREATE TABLE test (id INT)")
        assert "CREATE" in str(exc_info.value)

    def test_keyword_in_column_name_allowed(self):
        """Keywords as part of column names should NOT trigger blocking."""
        # These should pass - keywords are part of identifiers, not SQL commands
        valid_queries = [
            "SELECT UPDATED_AT FROM tblSales",
            "SELECT CREATED_DATE, DELETED_FLAG FROM tblSales",
            "SELECT order_insert_time FROM tblSales",
            "SELECT dropdown_value FROM tblSales",
        ]
        for query in valid_queries:
            validate_sql_query(query)  # Should not raise

    def test_case_insensitive_blocking(self):
        """Blocking should work regardless of case."""
        dangerous_queries = [
            "drop table users",
            "Drop Table Users",
            "DROP TABLE USERS",
            "DrOp TaBlE uSeRs",
        ]
        for query in dangerous_queries:
            with pytest.raises(SQLValidationError):
                validate_sql_query(query)

    def test_multiline_query_blocked(self):
        """Blocked keywords in multiline queries should be detected."""
        query = """
        SELECT * FROM users;
        DROP TABLE users;
        """
        with pytest.raises(SQLValidationError):
            validate_sql_query(query)


class TestSQLValidationIntegration:
    """Integration tests for SQL validation in ToolQueryLLMDB."""

    def test_tool_blocks_dangerous_query(self):
        """The tool should return an error for dangerous queries."""
        db = SampleOrder_LLM_DB()
        tqd = ToolQueryLLMDB(db)
        result_gen = tqd("DROP TABLE tblSales")
        for result in result_gen:
            pass
        assert "SQL code NOT executed" in result
        assert "DROP" in result

    def test_tool_blocks_update_query(self):
        """The tool should return an error for UPDATE queries."""
        db = SampleOrder_LLM_DB()
        tqd = ToolQueryLLMDB(db)
        result_gen = tqd("UPDATE tblSales SET price = 0")
        for result in result_gen:
            pass
        assert "SQL code NOT executed" in result
        assert "UPDATE" in result

    def test_tool_allows_valid_select(self, mock_duckdb_sql):
        """The tool should allow valid SELECT queries."""
        db = SampleOrder_LLM_DB()
        tqd = ToolQueryLLMDB(db)
        result_gen = tqd("SELECT * FROM tblSales LIMIT 5")
        for result in result_gen:
            pass
        assert "SQL code executed correctly" in result
