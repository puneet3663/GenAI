import json
import boto3
import snowflake.connector
import os

# Environment variables (BEST PRACTICE)
SNOWFLAKE_USER = os.environ['SF_USER']
SNOWFLAKE_PASSWORD = os.environ['SF_PASSWORD']
SNOWFLAKE_ACCOUNT = os.environ['SF_ACCOUNT']
SNOWFLAKE_WAREHOUSE = os.environ['SF_WAREHOUSE']
SNOWFLAKE_DATABASE = os.environ['SF_DATABASE']
SNOWFLAKE_SCHEMA = os.environ['SF_SCHEMA']

bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"


def get_snowflake_connection():
    return snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )


def generate_sql(user_query):
    prompt = f"""
You are a SQL expert.

Convert the following natural language query into a Snowflake SQL query.

### SCHEMA:
customers(customer_id, name, city)
orders(order_id, customer_id, order_date, amount)
order_items(order_id, product_id, quantity)
products(product_id, name, category)

### JOINS:
customers.customer_id = orders.customer_id
orders.order_id = order_items.order_id
order_items.product_id = products.product_id

### RULES:
- Only generate SELECT queries
- Use proper joins
- Limit results to 100 rows
- Do NOT add explanations, only SQL

User Query:
{user_query}
"""

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }),
        contentType="application/json",
        accept="application/json"
    )

    response_body = json.loads(response["body"].read())
    sql_query = response_body["content"][0]["text"]

    return sql_query.strip()


def validate_sql(sql):
    sql_lower = sql.lower()

    if not sql_lower.startswith("select"):
        raise Exception("Only SELECT queries are allowed")

    if "drop" in sql_lower or "delete" in sql_lower or "insert" in sql_lower:
        raise Exception("Unsafe query detected")

    return True


def execute_query(sql):
    conn = get_snowflake_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(sql)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()

        result = [dict(zip(columns, row)) for row in rows]
        return result

    finally:
        cursor.close()
        conn.close()


def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        user_query = body.get("query")

        if not user_query:
            return {
                "statusCode": 400,
                "body": "Query is required"
            }

        # Step 1: Generate SQL
        sql_query = generate_sql(user_query)

        # Step 2: Validate SQL
        validate_sql(sql_query)

        # Step 3: Execute
        results = execute_query(sql_query)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "sql": sql_query,
                "data": results
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": str(e)
        }
