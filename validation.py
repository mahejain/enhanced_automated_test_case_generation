import pandas as pd


def validate_row(row, domain_rules, conditional_vars=None):
    if conditional_vars is None:
        conditional_vars = []

    non_conditional_cols = [
        col for col in row.index
        if col not in conditional_vars + ['executed_requirements', 'executed_requirement_texts']
    ]

    if any(pd.isna(row[col]) for col in non_conditional_cols):
        return False

    for col, rule in domain_rules.items():
        if col in row and not pd.isna(row[col]) and not rule(row[col]):
            return False

    return True


def filter_valid_cases(df, domain_rules, conditional_vars=None):
    mask = df.apply(
        lambda row: validate_row(row, domain_rules, conditional_vars),
        axis=1
    )
    return df[mask].reset_index(drop=True)
