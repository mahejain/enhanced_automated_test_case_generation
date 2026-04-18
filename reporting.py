import pandas as pd
import logging
import numpy as np

logger = logging.getLogger(__name__)

def generate_requirement_wise_testcases(test_cases_df, equations, path="requirement_wise_testcases.csv"):

    df = test_cases_df.copy().reset_index(drop=True)
    df.insert(0, "test_case_no", df.index + 1)

    # CRITICAL FIX
    df["executed_requirements"] = df["executed_requirements"].str.split(",")

    exploded = df.explode("executed_requirements")

    rows = []

    for req in sorted(exploded["executed_requirements"].dropna().unique()):
        subset = exploded[exploded["executed_requirements"] == req]

        for _, row in subset.iterrows():
            rows.append([req] + row.tolist())

    columns = ["Requirement_ID"] + list(df.columns)

    result_df = pd.DataFrame(rows, columns=columns)
    result_df.to_csv(path, index=False)

    return result_df


def generate_requirement_wise_reports(test_cases_df, equations, ranges):

    df = test_cases_df.copy().reset_index(drop=True)
    df.insert(0, "test_case_no", df.index + 1)

    # SPLIT FIRST (same reason as above)
    df["executed_requirements"] = df["executed_requirements"].str.split(",")

    numeric_cols = list(ranges.keys())

    all_rows = []
    summary_lines = []

    for eq in equations:
        req = eq["id"]

        success_mask = df["executed_requirements"].apply(lambda x: req in x if isinstance(x, list) else False)

        success_df = df[success_mask].copy()
        failure_df = df[~success_mask].copy()

        # NEW EDGE LOGIC (data-driven)
        edge_mask = pd.Series(False, index=success_df.index)

        for col in numeric_cols:
            if col not in success_df.columns:
                continue

            col_min = df[col].min()
            col_max = df[col].max()

            edge_mask |= (success_df[col] == col_min) | (success_df[col] == col_max)

        edge_df = success_df[edge_mask]
        pure_success_df = success_df[~edge_mask]

        def add_rows(sub_df, label):
            if not sub_df.empty:
                temp = sub_df.copy()
                temp.insert(0, "Category", label)
                temp.insert(0, "Requirement_ID", req)
                all_rows.append(temp)

        add_rows(pure_success_df, "Success")
        add_rows(edge_df, "Edge")
        add_rows(failure_df, "Failure")

        summary_lines.append(
            f"{req}\n"
            f"  Success : {len(pure_success_df)}\n"
            f"  Edge    : {len(edge_df)}\n"
            f"  Failure : {len(failure_df)}\n"
        )

    final_df = pd.concat(all_rows, ignore_index=True)
    final_df.to_csv("requirement_wise_detailed.csv", index=False)

    with open("requirement_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))

    return final_df