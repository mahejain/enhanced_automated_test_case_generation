import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
import graphviz
import shutil
import os
import networkx as nx
import streamlit as st
import gravis as gv
import streamlit.components.v1 as components
# from few_shot_prompting import get_few_shot_prompting_response as get_few_shot_response
from prompting import get_few_shot_prompting_response as get_few_shot_response
from validation import filter_valid_cases
import random
import re 
from reporting import generate_requirement_wise_testcases
from reporting import generate_requirement_wise_reports
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


confusables = [
    ('^', '**'),
    ('acos', 'cos')
]

# ======================================================
# PREPROCESSING FUNCTIONS
# ======================================================

# remove confusables
def pre_process_equations(equations: list) -> None:
    for equation in equations:
        for confusable in confusables:
            equation['equation'] = equation['equation'].replace(confusable[0], confusable[1])
            if equation['condition']:
                equation['condition'] = equation['condition'].replace(confusable[0], confusable[1])

        # ---- FIX CONDITION FORMAT ----
        if equation['condition']:
            cond = equation['condition'].strip()

            # Remove leading "if"
            if cond.lower().startswith("if "):
                cond = cond[3:].strip()

            # Remove trailing "then"
            if cond.lower().endswith(" then"):
                cond = cond[:-5].strip()

            # Fix Enabled / Not_Enabled pattern
            cond = re.sub(
                r'(\w+)\s+Enabled',
                r'\1 == "enabled"',
                cond
            )

            cond = re.sub(
                r'(\w+)\s+Not_Enabled',
                r'\1 == "not_enabled"',
                cond
            )

            equation['condition'] = cond

# generate the list of values for each independant variables
def pre_process_ranges(ranges: dict, edge_cases_only: bool = True) -> dict:
    processed_ranges = {}
    for var, range_value in ranges.items():
        if isinstance(range_value[0], int) and isinstance(range_value[1], int):
            edge_size = 1
            step = range_value[2] if range_value[2] != 0 else 1  
            processed_ranges[var] = list(np.arange(range_value[0], range_value[1] + 1, step))
            if edge_cases_only:
                vals = processed_ranges[var]
                processed_ranges[var] = vals[:edge_size] + vals[-edge_size:]
        else:
            processed_ranges[var] = range_value
    return processed_ranges

# adding pre requisites required for calculating each dependant variable
def add_pre_requisites(equations, dependant_variables, variables):
    for equation in equations:
        equation['calculation_pre_requisites'] = []
        equation['calculation_pre_requisites_dependant'] = []
        equation['condition_pre_requisites'] = []
        equation['condition_pre_requisites_dependant'] = []
        rhs = equation['equation'].split('=')[1]
        equation['lhs'] = equation['equation'].split('=')[0].strip()
        condition = equation['condition']

        equation["dependant"] = condition is not None
        
        for var in variables:
            if re.search(rf'\b{var}\b', rhs):
                equation['calculation_pre_requisites'].append(var)
                if var in dependant_variables:
                    equation['calculation_pre_requisites_dependant'].append(var)
            if condition and re.search(rf'\b{var}\b', condition):
                equation['condition_pre_requisites'].append(var)
                if var in dependant_variables:
                    equation['condition_pre_requisites_dependant'].append(var)


def add_equation_conditions(equations):
    equations_dict = {}
    for equation in equations:
        lhs = equation['equation'].split('=')[0].strip()
        if lhs in equations_dict:
            equations_dict[lhs].append(equation)
        else:
            equations_dict[lhs] = [equation]
    return equations_dict


def check_cyclic_dependancy(equations_dict):
    visited = set()
    stack = set()
    def dfs(var):
        if var in stack:
            raise ValueError(f"Cyclic dependency detected involving '{var}'")
        if var in visited:
            return
        stack.add(var)
        for eq in equations_dict.get(var, []):
            for dep in eq['calculation_pre_requisites_dependant']:
                dfs(dep)
        stack.remove(var)
        visited.add(var)
    for variable in equations_dict:
        dfs(variable)


def topological_sort(dependant_variables, equations_dict):
    sorted_variables = []
    visited = set()

    def dfs(variable):
        if variable in visited:
            return
        visited.add(variable)

        equations = equations_dict.get(variable, [])

        for equation_info in equations:
            calculation_pre_requisites_dependant = equation_info['calculation_pre_requisites_dependant']

            for pre_req_variable in calculation_pre_requisites_dependant:
                dfs(pre_req_variable)

        sorted_variables.append(variable)

    for variable in dependant_variables:
        dfs(variable)

    return sorted_variables


def extract_equality_constraints(equations):
    constraints = {}
    for eq in equations:
        if eq.get('condition'):
            # Find patterns like 'var == value' (supports floats/ints)
            matches = re.findall(r'(\w+)\s*==\s*([0-9.-]+)', eq['condition'])
            for var, val in matches:
                try:
                    constraints[var] = float(val) if '.' in val else int(val)
                except ValueError:
                    pass  
    return constraints


def add_ranges_for_miscellaneous(
    variables,
    dependant_variables,
    independant_variables,
    constants,
):
    miscellaneous = [
        var for var in variables
        if var not in dependant_variables
        and var not in independant_variables
        and var not in constants
    ]

    if miscellaneous:
        error_msg = (
            f"Missing ranges for variables: {miscellaneous}. "
            f"Please define them explicitly in FRD input."
        )

        logger.error(error_msg)
        raise ValueError(error_msg)


# ======================================================
# SAMPLING FUNCTIONS
# ======================================================
def split_boundary_and_interior(ranges: dict):
    """
    Splits numeric ranges into:
    - boundary values (min, max)
    - interior values (everything in between)
    """
    boundary = {}
    interior = {}

    for var, r in ranges.items():

        # numeric range
        if isinstance(r, list) and len(r) == 3:
            start, end, step = r
            values = list(np.arange(start, end + step, step))

            if len(values) >= 2:
                boundary[var] = [values[0], values[-1]]
                interior[var] = values[1:-1] if len(values) > 2 else []
            else:
                boundary[var] = values
                interior[var] = []

        else:
            # categorical -> treat as boundary only
            boundary[var] = r
            interior[var] = []

    return boundary, interior


def monte_carlo_sampling(ranges: dict, n_samples: int = 100, constraints: dict = None, n_forced: int = 5) -> list:
    """
    Generate random samples, plus targeted ones to satisfy constraints.
    - constraints: dict of {var: value} to force.
    - n_forced: number of extra samples per constraint.
    """
    cached_ranges = {
        k: (np.arange(v[0], v[1] + v[2], v[2]) if isinstance(v, list) and len(v) == 3 else v)
        for k, v in ranges.items()
    }
    samples = []
    
    # Standard random samples
    for _ in range(n_samples):
        row = {}
        for var, r in ranges.items():
            if isinstance(r, list) and len(r) == 3:
                low, high, step = r
                values = cached_ranges[var]
                row[var] = random.choice(values)
            else:
                row[var] = random.choice(r)
        samples.append(row)
    
    # Add forced samples for each constraint
    if constraints:
        for var, forced_value in constraints.items():
            if var in ranges:
                for _ in range(n_forced):
                    row = {}
                    for v, r in ranges.items():
                        if v == var:
                            row[v] = forced_value  
                        else:
                            # Randomize others
                            if isinstance(r, list) and len(r) == 3:
                                low, high, step = r
                                values = cached_ranges[v]
                                row[v] = random.choice(values)
                            else:
                                row[v] = random.choice(r)
                    samples.append(row)
    
    return samples


def generate_monte_carlo_cases(ranges: dict, n_samples: int = 200) -> pd.DataFrame:
    """
    Additional random stress-test generator.
    Returns DataFrame directly.
    """

    samples = monte_carlo_sampling(ranges, n_samples)
    return pd.DataFrame(samples)


def hybrid_sampling(ranges: dict,
                    n_boundary: int = 50,
                    n_interior: int = 150):

    boundary_ranges, interior_ranges = split_boundary_and_interior(ranges)

    samples = []

    # ---- Boundary-focused sampling ----
    for _ in range(n_boundary):
        row = {}
        for var, vals in boundary_ranges.items():
            if vals:
                row[var] = random.choice(vals)
        samples.append(row)

    # ---- Interior sampling ----
    for _ in range(n_interior):
        row = {}
        for var in ranges.keys():
            if interior_ranges[var]:
                row[var] = random.choice(interior_ranges[var])
            else:
                # fallback to boundary if no interior
                row[var] = random.choice(boundary_ranges[var])
        samples.append(row)

    return pd.DataFrame(samples)
# ======================================================
# EVALUATION ENGINE
# ======================================================


# get statistics
def get_statistics(dependant_variables, test_cases_df):
    statistics = ''  # Initialize once
    for dependant_variable in dependant_variables:
        if dependant_variable not in test_cases_df.columns or test_cases_df[dependant_variable].isna().all():
            logger.info(f"Statistics for {dependant_variable}: Not calculated (condition not met or error)")
            continue
        try:
            statistics += f"Statistics for {dependant_variable}:\n"
            statistics += f"Minimum: {test_cases_df[dependant_variable].min()}\n"
            statistics += f"Maximum: {test_cases_df[dependant_variable].max()}\n"
            statistics += f"Median: {test_cases_df[dependant_variable].median()}\n"
            statistics += f"Unique values: {test_cases_df[dependant_variable].nunique()}\n\n"
        except Exception as e:
            logger.error(f"Error calculating statistics for {dependant_variable}: {e}")
    return statistics

# get styled HTML statistics
def get_styled_html_statistics(dependant_variables, test_cases_df):
    styled_statistics = '<style>table {border-collapse: collapse; width: 100%;} th, td {border: 1px solid #dddddd; text-align: left; padding: 8px;} th {background-color: #f2f2f2;}</style>'
    for dependant_variable in dependant_variables:
        try:
            styled_statistics += f"<h3>Statistics for {dependant_variable}</h3>"
            styled_statistics += "<table>"
            styled_statistics += f"<tr><td>Minimum</td><td>{test_cases_df[dependant_variable].min()}</td></tr>"
            styled_statistics += f"<tr><td>Maximum</td><td>{test_cases_df[dependant_variable].max()}</td></tr>"
            styled_statistics += f"<tr><td>Median</td><td>{test_cases_df[dependant_variable].median()}</td></tr>"
            styled_statistics += f"<tr><td>Unique values</td><td>{test_cases_df[dependant_variable].nunique()}</td></tr>"
            styled_statistics += "</table><br>"
        except Exception as e:
            logger.error(e)
    return styled_statistics


def show_variable_trend(dependant_variables, test_cases_df, streamlit=False):
    for variable in dependant_variables:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(test_cases_df[variable])
        ax.set_title(variable.capitalize())
        ax.set_xlabel(f'{variable} value')
        ax.set_ylabel('count')
        if streamlit:
            st.pyplot(fig)
        plt.close(fig)

# generate html
def generate_html(test_cases_df, path):
    test_cases_html = test_cases_df.to_html()
    with open(path, "w", encoding="utf-8") as text_file:
        text_file.write(test_cases_html)    

# identify dependancy trend
def identify_dependancy_trend(dependant_variables, equations_dict, test_cases_df, streamlit=False):
    try:
        num_dependant_variables = len(dependant_variables)
        calc_dependant_variables = {}
        num_variables = 0
        
        for var in dependant_variables:
            for equation in equations_dict[var]:
                if var in calc_dependant_variables:
                    calc_dependant_variables[var] += equation['calculation_pre_requisites']
                else: 
                    calc_dependant_variables[var] = equation['calculation_pre_requisites']
            calc_dependant_variables[var] = list(set(calc_dependant_variables[var]))
            num_variables = max(num_variables, len(calc_dependant_variables[var]))
                    
        fig, axs = plt.subplots(num_dependant_variables, num_variables, figsize=(5*num_variables, 6*num_dependant_variables))

        if num_dependant_variables == 1:
            axs = axs.reshape(1, num_variables)

        for j, dependant_variable in enumerate(dependant_variables):
            for i, independant_variable in enumerate(calc_dependant_variables[dependant_variable]):
                axs[j, i].scatter(test_cases_df[independant_variable], test_cases_df[dependant_variable])
                axs[j, i].set_title(f'{independant_variable.capitalize()} vs {dependant_variable.capitalize()}')
                axs[j, i].set_xlabel(f'{independant_variable.capitalize()}')
                axs[j, i].set_ylabel(f'{dependant_variable.capitalize()}')

        plt.tight_layout()
        if streamlit:
            st.pyplot(fig)
        plt.close()
    except Exception as e:
        logger.error(e)


def fill_dependant_variables(
    dependant_variables: list,
    equations_dict: dict,
    test_cases_df: pd.DataFrame,
    namespace: dict
) -> tuple:

    test_cases_df["executed_requirements"] = ""
    test_cases_df["executed_requirement_texts"] = ""

    coverage_counter = {}
    condition_hits = {}

    for index in test_cases_df.index:

        row_dict = test_cases_df.loc[index].to_dict()

        # Execute in already topologically sorted order
        for variable in dependant_variables:

            equations = equations_dict.get(variable, [])

            for eq in equations:

                calc_pre = eq['calculation_pre_requisites']
                cond_pre = eq['condition_pre_requisites']

                try:
                    # Skip if prerequisites missing
                    if any(pd.isna(row_dict.get(p)) for p in calc_pre + cond_pre):
                        continue

                    cond_ok = True

                    if eq['compiled_condition']:
                        try:
                            cond_result = eval(eq['compiled_condition'], namespace, row_dict)
                        except:
                            cond_result = False

                        condition_key_true = f"{eq['id']}_condition_true"
                        condition_key_false = f"{eq['id']}_condition_false"

                        if cond_result:
                            condition_hits[condition_key_true] = condition_hits.get(condition_key_true, 0) + 1
                        else:
                            condition_hits[condition_key_false] = condition_hits.get(condition_key_false, 0) + 1

                        cond_ok = cond_result

                    if cond_ok:
                        logger.debug(f"Processing {eq['id']} for {variable}, cond_ok: {cond_ok}")
                        logger.debug(f"Deps: {eq['calculation_pre_requisites']}")
                        for dep in eq['calculation_pre_requisites']:
                            logger.debug(f"  {dep}: in row_dict={dep in row_dict}, value={row_dict.get(dep)}, notna={pd.notna(row_dict.get(dep))}")
                        # Check if all calculation prerequisites are available and not NaN
                        if all(dep in row_dict and row_dict[dep] is not None and pd.notna(row_dict[dep]) for dep in eq['calculation_pre_requisites']):
                            logger.debug(f"Evaluating {eq['id']} for {variable}. row_dict keys: {list(row_dict.keys())}")
                            value = eval(eq['compiled_rhs'], namespace, row_dict)
                            
                            test_cases_df.at[index, variable] = value
                            row_dict[variable] = value 

                            req_id = eq["id"]
                            coverage_counter[req_id] = coverage_counter.get(req_id, 0) + 1

                            existing_ids = test_cases_df.at[index, "executed_requirements"]
                            existing_texts = test_cases_df.at[index, "executed_requirement_texts"]

                            test_cases_df.at[index, "executed_requirements"] = (
                                existing_ids + "," + req_id if existing_ids else req_id
                            )

                            test_cases_df.at[index, "executed_requirement_texts"] = (
                                existing_texts + " | " + eq["text"] if existing_texts else eq["text"]
                            )

                except Exception as e:
                    logger.error(f"Eval failed for {variable}: {e}")
                    continue

    return test_cases_df, condition_hits


def delete_directory_contents(directory_path):
    try:
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
        os.makedirs(directory_path)
        logger.info(f"Cleared directory: {directory_path}")
    except Exception as e:
        logger.error(e)
        

# ======================================================
# VISUALIZATION
# ======================================================


def create_flow_graph(directory_path, equations_dict):
    flowchart_groups = {}
    delete_directory_contents(directory_path)
    
    for var, equations in equations_dict.items():
        for equation in equations:
            pre_req_vars = equation['condition_pre_requisites']
            key = tuple(sorted(pre_req_vars + [var]))

            if key not in flowchart_groups:
                flowchart_groups[key] = []

            flowchart_groups[key].append((pre_req_vars, var, equation['equation'], equation['condition']))

    graph_index = 1
    for key, flowchart_data_group in flowchart_groups.items():
        flowchart_name = f'{directory_path}/{graph_index}'
        
        graph = graphviz.Digraph(flowchart_name, format='png') 

        for pre_req_vars, var, equation, condition in flowchart_data_group:
            graph.node(equation)
            for node_from in pre_req_vars:
                graph.node(node_from)
                graph.edge(node_from, equation, label=condition)

        graph.render(filename=flowchart_name, format='png', cleanup=True)
        graph_index += 1
        # flowchart_path = flowchart_name + '.png'


def create_new_flow_graph(flowchart_data, directory_path, equations_dict):
    print(flowchart_data, equations_dict)
    flowchart_groups = {}
    delete_directory_contents(directory_path)
    
    for var, equations in equations_dict.items():
        for equation in equations:
            pre_req_vars = equation['condition_pre_requisites']
            key = list(set(sorted(pre_req_vars + [var])))

            if var not in flowchart_groups:
                flowchart_groups[var] = []

            flowchart_groups[var].append((pre_req_vars, var, equation['equation'], equation['condition']))

    graph_index = 1
    flowchart_name = f'{directory_path}/{graph_index}'
    
    graph = graphviz.Digraph(flowchart_name, format='png')
    graph.attr('graph', rankdir='TB')

    graph_nodes = {}
    for var, flowchart_data_group in flowchart_groups.items():

        for pre_req_vars, var, equation, condition in flowchart_data_group:
            graph.node(equation)
            if var not in graph_nodes:
                graph_nodes[var] = []
            graph_nodes[var].append(equation)
            
        for pre_req_vars, var, equation, condition in flowchart_data_group:
            for node_from in pre_req_vars:
                if node_from not in graph_nodes:
                    graph.node(node_from)
                    graph_nodes[node_from] = [node_from]

                for node_from_store in graph_nodes[node_from]:
                    graph.edge(node_from_store, equation, label=condition)
    graph.render(filename=flowchart_name, format='png', cleanup=True)
    graph_index += 1
    # flowchart_path = flowchart_name + '.png'

# initialize edge and edge labels
def initialize_edge_and_labels(edges, edge_labels, equations_dict):
    for var, equations in equations_dict.items():
        for equation in equations:
            edges += [(pre_req_var, var, equation['condition'], equation['equation']) for pre_req_var in equation['condition_pre_requisites']]
            new_edge_labels = {(pre_req_var, var) : equation['condition'] for pre_req_var in equation['condition_pre_requisites']}
            edge_labels.update(new_edge_labels)    

# save to csv
def save_to_csv(test_cases_df, path):
    df = test_cases_df.copy()   # safer (don’t mutate original)
    df.reset_index(drop=True, inplace=True)
    df.insert(0, "test_case_no", df.index + 1)
    df.to_csv(path, index=False) 

# create interactive graph
def create_interactive_graph(edges, streamlit=False):
    # edge_label_index = 2

    G = nx.DiGraph(directed=True)

    for edge in edges:
        G.add_node(edge[0])
        G.add_node(edge[3])
        G.nodes[edge[3]]['color'] = 'red'

    for edge in edges:
        G.add_edge(edge[0], edge[3], label=edge[2])
        G.edges[edge[0], edge[3]].update({'label': edge[2]})

    fig = gv.d3(G, show_edge_label=True, edge_label_data_source='label',
        layout_algorithm_active=True,
        use_collision_force=True,
        collision_force_radius=70,
        edge_label_rotation=0,
        edge_label_size_factor=0.7,
        edge_label_font='monospace',
        zoom_factor=1.5,
        many_body_force_strength=10)
    
    if streamlit:
        components.html(fig.to_html(), height=600)
    
# create graph image
def create_graph_image(edges, edge_labels, streamlit=False):
    try:
        # edge_label_index = 2
        G = nx.DiGraph(directed=True)

        for edge in edges:
            G.add_node(edge[0])
            G.add_node(edge[1])
        for edge in edges:
            G.add_edge(edge[0], edge[1], label=edge[2])
            G.edges[edge[0], edge[1]].update({'label': edge[2]})
        
        pos = nx.spring_layout(G)
        options = {
            'node_color': 'yellow',
            'node_size': 300,
            'width': 1,
            'arrowstyle': '-|>',
            'arrowsize': 12,
            'labels' : {node: node for node in G.nodes()}
        }
        fig, ax = plt.subplots()
        nx.draw_networkx(G, pos, arrows=True, **options)
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_color='red'
        )
        if streamlit:
            st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        print(e)

# generate namespace
def initialize_namespace():
    math_functions = ['acos', 'asin', 'atan', 'atan2', 'ceil', 'copysign', 'cos', 'cosh', 'degrees', 'exp',
                      'fabs', 'floor', 'fmod', 'frexp', 'hypot', 'isfinite', 'isinf', 'isnan', 'ldexp', 'log',
                      'log10', 'modf', 'pow', 'radians', 'sin', 'sinh', 'sqrt', 'tan', 'tanh']

    namespace = {func: getattr(math, func) for func in math_functions}
    return namespace


def get_few_shot_prompting_response(FRD = ''):
    logger.info("Waiting for few-shot response")
    # output from few shot prompting
    dict_from_few_shot_prompting = get_few_shot_response(FRD)
    logger.debug(dict_from_few_shot_prompting)
    return dict_from_few_shot_prompting


# ======================================================
# COVERAGE & METRICS
# ======================================================


def generate_coverage_report(test_cases_df, condition_hits, all_req_ids, total_rows):

    logger.info("\n========== REQUIREMENT COVERAGE REPORT ==========")

    rows = []

    for req in sorted(all_req_ids):

        rows_covered = test_cases_df["executed_requirements"].str.contains(
            rf"\b{req}\b", regex=True, na=False
        ).sum()

        percent = (rows_covered / total_rows * 100) if total_rows else 0

        logger.info(f"{req:10} -> {rows_covered:5} rows  ({percent:.1f}%)")

        rows.append([req, rows_covered, percent])

    coverage_df = pd.DataFrame(
        rows,
        columns=["Requirement_ID", "Rows_Covered", "Coverage_%"]
    )

    missing = [
        r for r in all_req_ids
        if not test_cases_df["executed_requirements"].str.contains(rf"\b{r}\b", regex=True, na=False).any()
    ]

    if missing:
        logger.info("\n⚠ Uncovered Requirements:")
        for r in missing:
            logger.info(f"   {r}")
    else:
        logger.info("\n✅ All requirements covered!")

    logger.info("\n===== CONDITION COVERAGE =====")
    for cond, hits in condition_hits.items():
        percent = (hits / total_rows * 100) if total_rows else 0
        logger.info(f"{cond:25} -> {hits:5} hits ({percent:.1f}%)")

    # -------- NEW ADDITION --------
    return coverage_df


def compute_test_quality_metrics(df):

    total = len(df)

    exclude_cols = ["executed_requirements", "executed_requirement_texts"]
    data_cols = [c for c in df.columns if c not in exclude_cols]

    unique_rows = len(df[data_cols].drop_duplicates())
    uniqueness = unique_rows / total * 100

    edge_hits = 0
    for col in df.select_dtypes(include='number'):
        vals = df[col]
        edge_hits += ((vals == vals.min()) | (vals == vals.max())).sum()

    numeric_cols = df.select_dtypes(include='number')
    edge_ratio = edge_hits / (total * len(numeric_cols.columns)) * 100 if len(numeric_cols.columns) else 0

    variance = 0
    if not numeric_cols.empty:
        scaled = (numeric_cols - numeric_cols.mean()) / numeric_cols.std(ddof=0)
        variance = scaled.var().mean()

    stats_df = pd.DataFrame({
        "Metric": ["Total_cases", "Uniqueness_%", "Edge_usage_%", "Normalized_variance"],
        "Value": [total, uniqueness, edge_ratio, variance]
    })

    return stats_df


def generate_traceability_matrix(test_cases_df, path="traceability_matrix.csv"):
    """
    Creates Requirement Traceability Matrix (RTM)
    Maps each test case to requirement executed
    """
    trace_df = test_cases_df[[
        "executed_requirements",
        "executed_requirement_texts"
    ]].copy()

    trace_df.insert(0, "test_case_no", range(1, len(trace_df) + 1))

    trace_df.to_csv(path, index=False)

    logger.info("TRACEABILITY MATRIX GENERATED")
    logger.info("\n%s", trace_df.head())


# ======================================================
# VALIDATION RULE BUILDER
# ======================================================

def build_rules_from_ranges(ranges):
    """
    Automatically create validation rules from FRD ranges.
    Numeric  -> range check
    Categorical -> membership check
    """

    rules = {}

    for var, values in ranges.items():

        # categorical (strings list)
        if isinstance(values[0], str):
            allowed = set(values)
            rules[var] = lambda x, a=allowed: x in a

        # numeric range [start, end, step]
        elif isinstance(values, list) and len(values) == 3:
            start, end, _ = values
            rules[var] = lambda x, s=start, e=end: s <= x <= e

    return rules


# ======================================================
# MAIN DRIVER
# ======================================================


def extract_frd_details(FRD: str):
    dict_from_few_shot_prompting = get_few_shot_response(FRD=FRD)

    variables = dict_from_few_shot_prompting['variables']
    equations = dict_from_few_shot_prompting['equations']
    ranges = dict_from_few_shot_prompting['ranges']

    # 🔥 BUILD equation string from lhs + rhs (CRITICAL FIX)
    for i, eq in enumerate(equations):
        eq["equation"] = f"{eq['lhs']} = {eq['rhs']}"
        eq["id"] = f"REQ_{i+1}"
        eq["text"] = eq["equation"]

        if eq["lhs"] not in variables:
            variables.append(eq["lhs"])

        rhs_vars = re.findall(r'[A-Za-z_]\w*', eq["rhs"])
        for v in rhs_vars:
            if v not in variables and not v.isnumeric():
                variables.append(v)

    return variables, equations, ranges


def extract_constants(equations, variables, ranges):
    constants = {}
    namespace_for_eval = initialize_namespace()

    for eq in equations:
        lhs, rhs = eq['equation'].split('=')
        lhs = lhs.strip()
        rhs = rhs.strip()

        # constant = no variable in rhs
        if not any(re.search(rf'\b{var}\b', rhs) for var in variables):
            try:
                value = eval(rhs, namespace_for_eval)
                constants[lhs] = value
                eq["is_constant"] = True  
            except:
                eq["is_constant"] = False
        else:
            eq["is_constant"] = False

    # remove constants ONLY from ranges
    for const in constants:
        ranges.pop(const, None)

    return constants, ranges, equations  


def build_dependency_model(variables, equations, ranges, constants, edge_cases_only=True):

    independant_variables = list(ranges.keys())

    # Build equation dictionary FIRST
    equations_dict = add_equation_conditions(equations)

    # Now dependent variables = LHS variables
    dependant_variables = list(equations_dict.keys())

    add_ranges_for_miscellaneous(
        variables,
        dependant_variables,
        independant_variables,
        constants,
    )

    # Preprocessing
    pre_process_equations(equations)
    add_pre_requisites(equations, dependant_variables, variables)

    ranges = pre_process_ranges(ranges, edge_cases_only)

    # Detect cycles
    check_cyclic_dependancy(equations_dict)

    # Topological order
    dependant_variables = topological_sort(dependant_variables, equations_dict)

    # Compile expressions
    for eq_list in equations_dict.values():
        for eq in eq_list:
            rhs = eq['equation'].split('=')[1].strip()
            rhs = rhs.replace("\n", " ").strip()
            eq['compiled_rhs'] = compile(rhs, "<string>", "eval")

            if eq['condition']:
                eq['compiled_condition'] = compile(eq['condition'], "<string>", "eval")
            else:
                eq['compiled_condition'] = None

    variables = independant_variables + dependant_variables

    return variables, dependant_variables, equations_dict, ranges


def generate_input_samples(ranges, equations, limit=20):

    constraints = extract_equality_constraints(equations)

    # Small targeted base sampling (for equality conditions)
    base_samples = monte_carlo_sampling(
        ranges,
        n_samples=limit,
        constraints=constraints,
        n_forced=5
    )
    base_df = pd.DataFrame(base_samples)

    # Hybrid structured sampling
    hybrid_df = hybrid_sampling(
        ranges,
        n_boundary=80,
        n_interior=200
    )

    test_cases_df = pd.concat([base_df, hybrid_df], ignore_index=True)

    return test_cases_df


def execute_equations(test_cases_df, dependant_variables, equations_dict, constants):

    namespace = initialize_namespace()
    namespace.update(constants)

    # add constant columns first
    for const, val in constants.items():
        test_cases_df[const] = val

    # run normal evaluation FIRST (creates executed columns)
    test_cases_df, condition_hits = fill_dependant_variables(
        dependant_variables,
        equations_dict,
        test_cases_df,
        namespace
    )

    # NOW columns exist → safe to mark constants
    for eq_list in equations_dict.values():
        for eq in eq_list:
            if eq.get("is_constant"):
                test_cases_df["executed_requirements"] = (test_cases_df["executed_requirements"] + "," + eq["id"])

    test_cases_df["executed_requirements"] = (test_cases_df["executed_requirements"].fillna("").str.strip(",").apply(lambda x: ",".join(sorted(set(filter(None, x.split(",")))))))

    return test_cases_df, condition_hits


def generate_reports(test_cases_df, condition_hits, equations, dependant_variables, equations_dict, original_total_rows, ranges):

    all_req_ids = [eq["id"] for eq in equations]

    coverage_df = generate_coverage_report(
        test_cases_df, condition_hits, all_req_ids, original_total_rows
    )

    stats_df = compute_test_quality_metrics(test_cases_df)

    logger.info("\n========== STATISTICS REPORT ==========")
    logger.info("\n%s", stats_df.to_string(index=False))

    # Save directly in root
    coverage_df.to_csv("coverage_report.csv", index=False)
    stats_df.to_csv("statistics_report.csv", index=False)

    generate_traceability_matrix(test_cases_df, path="traceability_matrix.csv")

    save_to_csv(test_cases_df, path="test_cases_generated.csv")
    generate_html(test_cases_df, path="test_cases.html")

    statistics = get_statistics(dependant_variables, test_cases_df)
    logger.info("\n========== DEPENDANT VARIABLE STATS ==========")
    logger.info("\n%s", statistics)

    generate_requirement_wise_testcases(test_cases_df,equations, path="requirement_wise_testcases.csv")

    generate_requirement_wise_reports(test_cases_df,equations, ranges)

    with open("statistics.txt", "w") as f:
        f.write(statistics)

    create_flow_graph(
        directory_path='flowchart',   # keep this, it's intentional
        equations_dict=equations_dict
    )

    edges = []
    edge_labels = {}
    initialize_edge_and_labels(edges, edge_labels, equations_dict)
    create_interactive_graph(edges)

    logger.info("All reports saved in project root directory")


random.seed(42)
np.random.seed(42)


def generate_test_cases(FRD: str = '') -> pd.DataFrame:

    # Extract FRD Data
    variables, equations, ranges = extract_frd_details(FRD)

    # Extract Constants
    constants, ranges, equations = extract_constants(equations, variables, ranges)

    # Build Dependency Model
    variables, dependant_variables, equations_dict, ranges = build_dependency_model(
        variables,
        equations,
        ranges,
        constants,
        edge_cases_only=True
    )

    conditional_vars = [
        var for var, eq_list in equations_dict.items()
        if any(eq.get('condition') for eq in eq_list)
    ]
    logger.debug(f"Auto-detected conditional_vars: {conditional_vars}")

    #Generate Input Samples
    test_cases_df = generate_input_samples(ranges, equations, limit=20)

    # Execute Equations
    test_cases_df, condition_hits = execute_equations(
        test_cases_df,
        dependant_variables,
        equations_dict,
        constants
    )

    original_total_rows = len(test_cases_df)

    # Build validation rules automatically from FRD ranges
    domain_rules = build_rules_from_ranges(ranges)
    
    # Filter Valid Cases
    test_cases_df = filter_valid_cases(test_cases_df, domain_rules, conditional_vars)

    # Reporting
    generate_reports(test_cases_df, condition_hits, equations, dependant_variables, equations_dict, original_total_rows, ranges)

    return test_cases_df
    

if __name__ == '__main__':
    # input given 
    FRD = '''
    Auto_Drift_Estimate - [enabled, not_enabled]
    a0 - [1, 2, 1]
    a1 - [2, 2, 1]
    a2 - [3, 5, 1]
    a3 - [0, 2, 1]
    a4 - [0, 1, 1]
    a5 - [1, 2, 1]
    a6 - [0, 2, 1]
    actual_frequency - [7000, 7010, 3]
    present_temperature - [-10, 50, 8]
    t0 - [50, 100, 20]
    _RANGE_END_TAG_

    If Auto_Drift_Estimate Enabled
    If -10 < present_temperature < 55 then
    delfbyf0 = a0 + a1 * (present_temperature - temp0) + a2 * (present_temperature - temp0)^2 + a3 * (present_temperature - temp0)^3 + a4 * (present_temperature - temp0)^4 + a5 * (present_temperature - temp0)^5 + a6 * (present_temperature - temp0)^6
    Where
    f0 = (1/0.000125)
    theoretical_frequency = f0
    deltaf = actual_frequency - theoretical_frequency
    temp0 = 25 
    The above logic shall be computed every 7 sec as the temperature reading shall happen every 7 sec.
    deltaf = f0 * delfbyf0
    new_time_period = 1/(f0 + deltaf)
    delta_time_period = new_time_period - t0
    '''
    generate_test_cases(FRD)