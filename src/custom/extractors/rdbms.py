from sqlalchemy import text

class RDBMSExtractor:
    def __init__(self, connection, config):
        """ 
        connection: connection object
        config: what kind of data we have to extract
        """
        self.connection = connection
        self.config = config

    def __call__(self):
        return self.extract()

    def extract(self):
        results = {}
        # Get the list of tables from YAML
        tables = self.config.get("tables", [])

        for table in tables:
            # Get values dynamically from YAML
            name = table.get("table_name")
            schema = table.get("schema", "public") # Default to public if missing
            cols = table.get("columns")

            # Handle column selection
            column_query = "*" if not cols else ", ".join(cols)

            # Build the dynamic query
            query = f"SELECT {column_query} FROM {schema}.{name}"
            print(f"Executing: {query}")

            try:
                # Native SQLAlchemy execution
                result_proxy = self.connection.execute(text(query))
                
                # Convert to JSON-ready list of dicts
                rows = [dict(row) for row in result_proxy.mappings()]
                results[name] = rows
            
            except Exception as e:
                print(f"Error extracting {schema}.{name}: {e}")
                raise e
        
        return results