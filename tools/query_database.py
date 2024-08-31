"""
class llm_db has to have

 - description of all tables
 - description of all columns of all tables
 - JOIN columns (usually IDs)

the implementation will probably involve a WHERE with the account_id

so llm_db implements

llm_db.sql_query(q)

and does something like

query = WITH tbl_name as
(
	SELECT * FROM real_tbl
	WHERE account_id = (replace with account_id)
)

{q}

q will SELECT FROM tbl_name

in the LLM tool itself, possible arguments:

table: table to select from
columns_without_aggregation: list of columns, separated by commas, desired in the output whose contents will not be aggregated
aggregated_columns: list of columns, separated by commas, that will be aggregated
aggregation_functions: list of aggregation functions, separated by commas, that will be applied to each column of aggregated_columns.
	For example: MIN , MAX , SUM , AVG , COUNT , VARIANCE , and STDDEV

the output column names will be [original_column]_[aggregation]

selection_functions: argument to SQL WHERE. It can involve any column in the table and not just columns_without_aggregation and aggregated_columns.

having_functions: argument to SQL HAVING. It can involve any column in the table and the columns after aggregation in the form [original_column]_[aggregation].

sort_by: argument to SQL SORT BY

max_results: integer N to use in "LIMIT N" at the end
"""
