import sql_engine


# ============================================================
# SQL ENGINE COMPLETE TEST
# ============================================================

def main():

    print("=" * 70)
    print("SQL ENGINE - COMPLETE TEST")
    print("=" * 70)
    print()


    # ========================================================
    # 1. CHECK MODULE IMPORT
    # ========================================================

    print("[1] Importing SQL engine...")

    print("✅ sql_engine imported successfully.")
    print()


    # ========================================================
    # 2. SHOW AVAILABLE FUNCTIONS
    # ========================================================

    print("[2] Checking SQL engine functions...")

    functions = [
        name
        for name in dir(sql_engine)
        if not name.startswith("_")
        and callable(getattr(sql_engine, name))
    ]

    for function in functions:
        print(f"  - {function}")

    print()


    # ========================================================
    # 3. CHECK BIGQUERY CONNECTION
    # ========================================================

    print("[3] Checking BigQuery connection...")

    try:

        client = sql_engine.bq_client

        query = """
            SELECT 1 AS test
        """

        result = list(
            client.query(query).result()
        )

        if result and result[0].test == 1:

            print("✅ BigQuery connection works.")

        else:

            print("❌ BigQuery test returned unexpected result.")
            return

    except Exception as e:

        print("❌ BigQuery connection failed.")
        print()
        print(type(e).__name__)
        print(e)

        return

    print()


    # ========================================================
    # 4. TEST QUESTION
    # ========================================================

    question = (
        "Which city generated the highest revenue?"
    )

    print("[4] Test question:")
    print()
    print(question)
    print()


    # ========================================================
    # 5. GENERATE SQL
    # ========================================================

    print("[5] Generating SQL...")

    try:

        sql = sql_engine.generate_sql(
            question
        )

        print("✅ SQL generated successfully.")
        print()
        print("-" * 70)
        print(sql)
        print("-" * 70)

    except Exception as e:

        print("❌ SQL generation failed.")
        print()
        print(type(e).__name__)
        print(e)

        return

    print()


    # ========================================================
    # 6. EXECUTE SQL
    # ========================================================

    print("[6] Executing generated SQL...")

    try:

        result = sql_engine.run_query(
            sql
        )

        print("✅ SQL executed successfully.")
        print()


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        print("-" * 70)
        print("QUERY RESULT")
        print("-" * 70)

        print()


        if result is None:

            print("No result returned.")

        elif hasattr(result, "to_string"):

            print(
                result.to_string(
                    index=False
                )
            )

        else:

            print(result)


    except Exception as e:

        print("❌ SQL execution failed.")
        print()
        print(type(e).__name__)
        print(e)

        return

    print()


    # ========================================================
    # 7. COMPLETE TEST
    # ========================================================

    print("=" * 70)
    print("SQL ENGINE TEST COMPLETED")
    print("=" * 70)

    print()

    print("✅ sql_engine import works")
    print("✅ BigQuery connection works")
    print("✅ SQL generation works")
    print("✅ SQL execution works")

    print()


if __name__ == "__main__":
    main()