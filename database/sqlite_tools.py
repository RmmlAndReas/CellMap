"""SQLite utility functions for CellMap."""
import sqlite3


def get_table_columns(path_to_db, table_name, force_lower_case=False):
    """
    Get the column names from a SQLite table.

    Args:
        path_to_db (str): The path to the SQLite database file.
        table_name (str): The name of the table.
        force_lower_case (bool): Whether to force column names to lowercase.

    Returns:
        list: List of column names.
    """
    # Connect to the database
    conn = sqlite3.connect(path_to_db)

    # Create a cursor object
    cur = conn.cursor()

    # Execute a query to get the column names
    cur.execute("PRAGMA table_info({})".format(table_name))

    # Fetch the results and extract column names
    cols = []
    results = cur.fetchall()
    for row in results:
        col_name = row[1].strip("'")
        if force_lower_case:
            col_name = col_name.lower()
        cols.append(col_name)

    conn.close()
    return cols


__all__ = ['get_table_columns']
