# Thanks to https://www.kaggle.com/datasets/kyanyoga/sample-sales-data
# Sample CSV from that page (adjusted)
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
from abc import ABC, abstractmethod

import duckdb


class LLM_Database(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_database_name(self):
        """This method should return the name of the database"""
        pass

    @abstractmethod
    def get_database_info(self):
        """This method should provide a comprehensive description of the
        tables in the database as they will be accessible using the
        self.sql_query method, and any specifics about those tables and columns
        """
        pass

    @abstractmethod
    def get_tables(self):
        """This method should return a list of strings. Each contains one
        valid table name that can be used in the query (see self.sql_query)
        """
        pass

    @abstractmethod
    def sql_query(self, query):
        """This method should *safely* run queries in the database.
        Make sure that queries run by this method cannot change the contents
        of the database or do anything harmful. This method should also implement
        safeguards like pre-filtering per account id.

        A usual way of doing it would be:
        WITH exposed_table AS
        (
            SELECT * FROM real_table
            WHERE real_table.account_id = <ACCOUNT_ID>
        )
        [Insert user query that uses exposed_table]
        """
        pass

    def get_full_database_description(self):
        description = [
            "Information about the tables in the dataset can be found in the <database_tables></database_tables>:"
        ]
        description = ["<database_tables>"]
        for tbl in self.get_tables():
            df = self.sql_query(f"SELECT * FROM '{tbl}' LIMIT 5")
            description.append("<database_table>")
            description.append(f"<table_name>{tbl}</table_name>")

            # information
            description.append(
                f"<table_columns>{','.join(list(df.columns))}</table_columns>"
            )
            description.append(
                f"<table_column_types>{','.join([str(x) for x in list(df.dtypes)])}</table_column_types>"
            )
            xml_sample = df.to_xml(
                index=False, xml_declaration=False, root_name=f"table_sample_data"
            )
            description.append(xml_sample)
            description.append("</database_table>")
        description.append("</database_tables>")
        description.append(
            "Note that the information in <table_sample_data></table_sample_data> is just a very small sample of the records in the table. To retrieve real data, it is necessary to actually query the table."
        )
        description.append(self.get_database_info())
        return "\n".join(description)


class SampleOrder_LLM_DB(LLM_Database):
    def __init__(self):
        self.db_path = f"gat_llm/tools/query_database_sales_data_sample.csv"

    def get_database_name(self):
        return "Sales_database"

    def get_tables(self):
        return ["tblSales"]

    def sql_query(self, query, max_desired_results=400):
        # handle when the LLM decides to use WITH
        if len(query) > 4 and query[0:4].lower() == "with":
            query = ", " + query[5:]

        # handle LIMIT
        query_lines = [
            x for x in query.replace(";", "").splitlines() if x.strip() != ""
        ]
        if "limit " not in query_lines[-1].lower():
            query = query.replace(";", "") + f"\nLIMIT {max_desired_results + 1}"
        else:
            assert (
                int(query_lines[-1].split()[-1]) <= max_desired_results
            ), f"Error: the LIMIT clause cannot request for more than {max_desired_results} results"

        q = f"""
WITH tblSales AS
(
    SELECT * FROM '{self.db_path}'
)
{query}
"""
        return duckdb.sql(q).df()

    def get_database_info(self):
        return f"In table tblSales: Column PRODUCTCODE is a unique identifier of the product. ORDERNUMBER is a unique identifier of the order."


class ToolQueryLLMDB:
    def __init__(self, LLM_Database, max_records=100):
        self.db = LLM_Database
        self.max_records = max_records
        self.name = f"query_database_{self.db.get_database_name()}"

        db_description = self.db.get_full_database_description()

        self.tool_description = {
            "name": self.name,
            "description": f"""Generates SQL code that will be used to query the database {self.db.get_database_name()}.
If the query is too complicated, involving multiple JOINs or conditions, make sure to ask the user if the results make sense.
The details of the database are as follows:

{db_description}

Never use any commands that can modify the tables or records in the database.
If the number of records exceeds {max_records}, inform that to the user and help him refine the query to narrow down the search.
If the query is executed successfully, a table in XML format containing the results will be returned.
If an error happens, the error description will be returned.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sql_code": {
                        "type": "string",
                        "description": """Valid SQL code to query the database.""",
                    },
                },
                "required": ["sql_code"],
            },
        }

    def __call__(self, sql_code, **kwargs):
        if len(kwargs) > 0:
            yield f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"
            return

        try:
            yield f"<scratchpad>Executing: {sql_code}</scratchpad>"
            ans = self.db.sql_query(sql_code, max_desired_results=self.max_records)
            if len(ans) > self.max_records:
                final_ans = [
                    "SQL code NOT executed. Too many records. Please refine the search.",
                    f"Number of records found: {len(ans)}. Maximum allowed: {self.max_records}",
                ]
            else:
                final_ans = [
                    "SQL code executed correctly. Results:",
                    ans.to_xml(
                        index=False, xml_declaration=False, root_name=f"query_results"
                    ),
                ]
        except Exception as ex:
            final_ans = ["SQL code NOT executed. Error description:", str(ex)]

        yield "\n".join(final_ans)
