import pandas as pd
import networkx as nx

from typing import List, Tuple
from itertools import combinations
import re


DEFAULT_CHILDREN_ISSUES_LEFTCOL = False
DEFAULT_CHILDREN_ISSUES_RIGHTCOL = False


class Cols:
    def __init__(self, df: pd.DataFrame):
        self.col1 = str(df.columns[0])
        self.col2 = str(df.columns[1])

    @property
    def col_names(self) -> tuple[str, str]:
        return self.col1, self.col2


class Reference:
    def __init__(self, id: str, row_idx: int):
        self._id = str(id)
        self.row_idx = row_idx
        self._n_root_parts = 1
        self._formatted = self.clean_id()

    @property
    def id(self) -> str:
        return self._id

    # if id is changed later
    @id.setter
    def id(self, value: str) -> None:
        self._id = str(value)
        self._formatted = self.clean_id()

    @property
    def formatted(self) -> str:
        return self._formatted

    def clean_id(self) -> str:
        token_pattern = re.compile(r"[A-Za-z0-9]+")
        pieces = token_pattern.findall(self._id)
        return "_".join(pieces)

    @property
    def level(self) -> int:
        return len(self.formatted.split(sep="_"))


def create_relationships(df: pd.DataFrame, cols: Cols) -> None:
    col1, col2 = cols.col_names

    # degree of nodes
    degree_1 = df.groupby(col1).size()
    degree_2 = df.groupby(col2).size()
    # attaching to df
    df[f"degree_{col1}"] = df[col1].map(degree_1)
    df[f"degree_{col2}"] = df[col2].map(degree_2)

    # relationship labels
    def relation_label(row):
        num_A = row[f"degree_{col1}"]
        num_B = row[f"degree_{col2}"]
        if num_A == 1 and num_B == 1:
            return "one-to-one"
        elif num_A > 1 and num_B == 1:
            return "one-to-many"
        elif num_A == 1 and num_B > 1:
            return "many-to-one"
        else:
            return "many-to-many"

    df["relation"] = df.apply(lambda row: relation_label(row), axis=1)
    df.drop(columns=[f"degree_{col1}", f"degree_{col2}"], inplace=True)
    # dropping degree columns (not useful)


def create_graph_from_df(df: pd.DataFrame, cols: Cols) -> nx.Graph:
    col1, col2 = cols.col_names
    G = nx.Graph()

    # adding nodes and edges
    for _, row in df.iterrows():
        G.add_edge(f"{col1}_{row[col1]}", f"{col2}_{row[col2]}")

    # adding "type" attribute to nodes
    G.add_nodes_from([f"{col1}_{i}" for i in set(df[col1])], type=col1)
    G.add_nodes_from([f"{col2}_{i}" for i in set(df[col2])], type=col2)

    return G


def is_many_to_many_component(component, graph: nx.Graph, cols: Cols) -> bool:
    """Check if a component maintains many-to-many relationships."""
    col1, col2 = cols.col_names
    type_1 = [n for n in component if graph.nodes[n].get("type") == col1]
    type_2 = [n for n in component if graph.nodes[n].get("type") == col2]

    # Many-to-many: multiple nodes on each side
    return len(type_1) > 1 and len(type_2) > 1


def return_critical_edges_from_graph(graph) -> List[Tuple[str, str]]:
    return [
        (u, v)
        for u, v in graph.edges()
        if graph.edges[u, v].get("criticality_type") == "independent_critical"
    ]


def find_critical_edge_combinations(
    graph,
    cols: Cols,
    non_critical_bridges: List[Tuple],
    max_combination_size: int = 3,
) -> dict:
    """
    Find combinations of edges whose joint removal breaks the many-to-many structure.

    Args:
        component: Set of nodes in the component
        graph: NetworkX graph
        cols: Column names object
        non_critical_bridges: List of bridges that aren't individually critical
        max_combination_size: Maximum combination size to test (default 3 to avoid exponential blowup)

    Returns:
        Dictionary mapping frozenset of edges to effect description
    """
    combinations_dict: dict[frozenset, dict] = {}

    # Test combinations of size 2 and up
    for combo_size in range(
        2, min(max_combination_size + 1, len(non_critical_bridges) + 1)
    ):
        for edge_combo in combinations(non_critical_bridges, combo_size):
            # Create test graph and remove edge combination
            graph_test = graph.copy()
            for u, v in edge_combo:
                if graph_test.has_edge(u, v):
                    graph_test.remove_edge(u, v)

            new_comps = list(nx.connected_components(graph_test))

            # Check if combination breaks all M-N patterns
            if not any(
                is_many_to_many_component(c, graph_test, cols) for c in new_comps
            ):
                combinations_dict[frozenset(edge_combo)] = {
                    "breaks_m2n": True,
                    "description": f"Removing edges {len(edge_combo)}: {edge_combo} breaks M-N",
                }

        if combinations_dict:
            break

    return combinations_dict


def update_graph_with_attributes(components, graph, cols: Cols) -> dict[int, dict]:
    """
    Edge attributes assigned:
        - criticality_type: "independent_critical", "non_critical_bridge", or "cycle"
        - position_in_path: "dead_end_bridge", "interior_bridge", or "unknown"
        - critical_combinations: If edge is part of a multi-edge critical set, the set of edges that together are critical
        - component_id: Integer ID for the component this edge belongs to (all components)

    Returns:
        Dictionary mapping component_id (int) to component info dict with keys:
        - "nodes": set of node names in component
        - "edges": list of edges in component
        - "criticality_type": criticality classification for the component
    """
    component_map: dict[int, dict] = {}

    for current_comp_id, component in enumerate(components):
        is_m2m = is_many_to_many_component(component, graph, cols)
        component_graph = nx.subgraph(graph, component).copy()

        subgraph_edges = list(component_graph.edges)
        component_bridges = list(nx.bridges(component_graph))

        for u, v in subgraph_edges:
            graph.edges[u, v]["component_id"] = current_comp_id

        # Initialize component info
        component_map[current_comp_id] = {
            "nodes": component,
            "edges": subgraph_edges,
            "is_many_to_many": is_m2m,
            "criticality_type": None,  # Will be updated below
        }

        if not is_m2m:
            component_map[current_comp_id]["criticality_type"] = "not_many_to_many"
            continue

        if not component_bridges:
            # All edges are in cycles — no single critical edge possible
            component_map[current_comp_id]["criticality_type"] = "cycle"
            for edge in subgraph_edges:
                graph.edges[edge]["criticality_type"] = "cycle"
            continue

        individual_critical = []
        non_critical = []

        # Phase 1: Test individual bridges
        for u, v in subgraph_edges:
            deg_u = graph.degree(u)
            deg_v = graph.degree(v)

            # Classify by position in path
            if deg_u == 1 or deg_v == 1:
                position = "dead_end_bridge"
            elif deg_u == 2 and deg_v == 2:
                position = "interior_bridge"
            else:
                position = "higher-level"

            graph.edges[u, v]["position_in_path"] = position

            # Test criticality
            graph_test = component_graph.copy()
            graph_test.remove_edge(u, v)
            new_comps = list(nx.connected_components(graph_test))

            is_crit = not any(
                is_many_to_many_component(c, graph_test, cols) for c in new_comps
            )

            if is_crit:
                individual_critical.append((u, v))
                graph.edges[u, v]["criticality_type"] = "independent_critical"
                component_map[current_comp_id]["criticality_type"] = (
                    "has_critical_edges"
                )
            else:
                non_critical.append((u, v))
                graph.edges[u, v]["criticality_type"] = "non_critical_bridge"

        # Phase 2: Test combinations of non-critical bridges
        if non_critical and not individual_critical:
            critical_combos = find_critical_edge_combinations(
                component_graph, cols, non_critical, max_combination_size=3
            )

            if critical_combos:
                component_map[current_comp_id]["criticality_type"] = (
                    "has_critical_combos"
                )
                for edge_set, _ in critical_combos.items():
                    edges_list = list(edge_set)
                    # Mark edges as part of combination
                    for u, v in edges_list:
                        if "critical_combinations" not in graph.edges[u, v]:
                            graph.edges[u, v]["critical_combinations"] = []
                        graph.edges[u, v]["critical_combinations"].append(edges_list)
            else:
                component_map[current_comp_id]["criticality_type"] = "complex_cyclic"

    return component_map


def get_edgesCategories_map_from_attribute(
    graph, attr: str
) -> dict[str, List[Tuple[str, str]]]:
    mapping: dict[str, List[Tuple[str, str]]] = {}

    for u, v in graph.edges():
        category = graph.edges[u, v].get(attr)
        if category is None:
            continue

        # Keep keys hashable even when edge attributes are lists/dicts.
        try:
            hash(category)
            key = category
        except TypeError:
            key = str(category)

        if key not in mapping:
            mapping[key] = []
        mapping[key].append((u, v))

    return mapping


def apply_mapping_to_df(
    df: pd.DataFrame,
    cols: Cols,
    attr: str,
    mapping: dict[str, List[Tuple[str, str]]],
) -> None:
    """
    Create/update column `attr` assigning each dataframe edge to its category.

    If an edge is not present in `mapping`, the resulting value is NaN.
    """
    col1, col2 = cols.col_names

    # Build fast lookup: both directions map to same category.
    edge_to_category: dict[Tuple[str, str], str] = {}
    for cat, edges in mapping.items():
        for u, v in edges:
            edge_to_category.setdefault((u, v), cat)
            edge_to_category.setdefault((v, u), cat)

    df[attr] = df.apply(
        lambda row: edge_to_category.get(
            (f"{col1}_{row[col1]}", f"{col2}_{row[col2]}"), None
        ),
        axis=1,
    )


def _strip_node_prefix(node: object, cols: Cols) -> str:
    """Remove graph-side column prefix from a node label, when present."""
    col1, col2 = cols.col_names
    text = str(node)

    prefix_1 = f"{col1}_"
    prefix_2 = f"{col2}_"

    if text.startswith(prefix_1):
        return text[len(prefix_1) :]
    if text.startswith(prefix_2):
        return text[len(prefix_2) :]

    return text


def _sanitize_critical_combinations(value: object, cols: Cols) -> object:
    """Convert combo edge tuples from graph labels to raw mapping values."""
    if not isinstance(value, list):
        return value

    sanitized: list[list[tuple[str, str]]] = []
    for combo in value:
        if not isinstance(combo, (list, tuple)):
            continue

        cleaned_combo: list[tuple[str, str]] = []
        for edge in combo:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                continue

            left = _strip_node_prefix(edge[0], cols)
            right = _strip_node_prefix(edge[1], cols)
            cleaned_combo.append((left, right))

        if cleaned_combo:
            sanitized.append(cleaned_combo)

    return sanitized if sanitized else pd.NA


def apply_edge_attribute_to_df(df: pd.DataFrame, cols: Cols, attr: str, graph) -> None:
    """Copy raw edge attribute values from graph into dataframe column `attr`."""
    col1, col2 = cols.col_names
    edge_to_value: dict[Tuple[str, str], object] = {}

    for u, v in graph.edges():
        value = graph.edges[u, v].get(attr, pd.NA)
        if attr == "critical_combinations":
            value = _sanitize_critical_combinations(value, cols)
        edge_to_value[(u, v)] = value
        edge_to_value[(v, u)] = value

    df[attr] = df.apply(
        lambda row: edge_to_value.get(
            (f"{col1}_{row[col1]}", f"{col2}_{row[col2]}"), pd.NA
        ),
        axis=1,
    )


def check_relationships(df: pd.DataFrame):
    if "position_in_path" not in df.columns or "relation" not in df.columns:
        return

    df.loc[df["position_in_path"] == "dead_end_bridge", "relation"] = "many-to-many"


def apply_component_id_to_df(df: pd.DataFrame, cols: Cols, graph) -> None:
    """Apply component_id to all rows using component IDs already set on graph edges."""
    col1, col2 = cols.col_names

    # Build edge-to-component mapping from node mapping
    edge_to_comp_id: dict[Tuple[str, str], int] = {}
    for u, v in graph.edges():
        edge_comp_id = graph.edges[u, v].get("component_id")
        if edge_comp_id is not None:
            edge_to_comp_id[(u, v)] = edge_comp_id
            edge_to_comp_id[(v, u)] = edge_comp_id

    df["component_id"] = df.apply(
        lambda row: edge_to_comp_id.get(
            (f"{col1}_{row[col1]}", f"{col2}_{row[col2]}"), pd.NA
        ),
        axis=1,
    )


def return_childparents_issues(
    df: pd.DataFrame,
    cols: Cols,
    enabled_col1: bool,
    enabled_col2: bool,
) -> pd.DataFrame:
    col1, col2 = cols.col_names
    DP_cols: pd.DataFrame = df[[col1, col2, "component_id"]].copy()
    DP_cols["__row_id__"] = range(len(DP_cols))

    # Track which rows are problematic for each column
    problematic_rows_col1: set = set()
    problematic_rows_col2: set = set()

    if enabled_col1:
        # First pass: col1 is focus (checking children in col1 grouped by col2)
        for _, group in DP_cols.groupby(["component_id", col2], dropna=False):
            analyse_childparents_issues(
                group[col1], problematic_rows_col1, row_ids=group["__row_id__"]
            )

    if enabled_col2:
        # Second pass: col2 is focus (checking children in col2 grouped by col1)
        for _, group in DP_cols.groupby(["component_id", col1], dropna=False):
            analyse_childparents_issues(
                group[col2], problematic_rows_col2, row_ids=group["__row_id__"]
            )

    # Add columns to original dataframe
    df[f"is_problematic_child_{col1}"] = (
        DP_cols["__row_id__"].isin(problematic_rows_col1).to_numpy()
    )
    df[f"is_problematic_child_{col2}"] = (
        DP_cols["__row_id__"].isin(problematic_rows_col2).to_numpy()
    )
    return df


def analyse_childparents_issues(
    series: pd.Series,
    problematic_rows: set,
    row_ids: pd.Series | None = None,
) -> None:
    """Mark problematic row indices in the set (side effect)."""
    lookup_group = []
    stable_row_ids = row_ids.tolist() if row_ids is not None else series.index.tolist()

    for stable_row_id, element in zip(stable_row_ids, series.tolist()):
        reference = Reference(element, stable_row_id)
        lookup_group.append(reference)

    token_paths: dict[int, tuple[str, ...]] = {}
    existing_paths: set[tuple[str, ...]] = set()

    for ref in lookup_group:
        tokens = tuple(part for part in ref.formatted.split("_") if part)
        token_paths[ref.row_idx] = tokens
        existing_paths.add(tokens)

    def has_existing_parent(path: tuple[str, ...]) -> bool:
        if len(path) <= 1:
            return False

        direct_parent = path[:-1]
        if direct_parent in existing_paths:
            return True

        return has_existing_parent(direct_parent)

    for row_idx, path in token_paths.items():
        if has_existing_parent(path):
            problematic_rows.add(row_idx)


def main(
    df: pd.DataFrame,
    children_issues_leftcol: bool = DEFAULT_CHILDREN_ISSUES_LEFTCOL,
    children_issues_rightcol: bool = DEFAULT_CHILDREN_ISSUES_RIGHTCOL,
) -> tuple[pd.DataFrame, nx.Graph, dict[int, dict]]:

    cols = Cols(df)

    # relationship labels (one-to-one, one-to-many, many-to-one, many-to-many)
    create_relationships(df, cols)

    # NetworkX graph
    G = create_graph_from_df(df, cols)

    # Analyze graph to find critical edges and combinations
    component_map = update_graph_with_attributes(nx.connected_components(G), G, cols)

    criticality_types = get_edgesCategories_map_from_attribute(G, "criticality_type")
    position_in_path = get_edgesCategories_map_from_attribute(G, "position_in_path")

    apply_mapping_to_df(df, cols, "criticality_type", criticality_types)
    apply_mapping_to_df(df, cols, "position_in_path", position_in_path)
    apply_edge_attribute_to_df(df, cols, "critical_combinations", G)
    apply_component_id_to_df(df, cols, G)

    check_relationships(df)

    col1, col2 = cols.col_names
    if not children_issues_leftcol and not children_issues_rightcol:
        df[f"is_problematic_child_{col1}"] = False
        df[f"is_problematic_child_{col2}"] = False
        return df, G, component_map

    df = return_childparents_issues(
        df,
        cols,
        enabled_col1=children_issues_leftcol,
        enabled_col2=children_issues_rightcol,
    )

    return df, G, component_map
