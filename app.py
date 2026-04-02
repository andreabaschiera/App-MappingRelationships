import pandas as pd
import streamlit as st
import networkx as nx
from html import escape

from analyse import main, Cols

st.set_page_config(page_title="Mapping Analysis", layout="wide")
st.title("Interoperability: Relationships")
st.markdown(
    "Analyze mapping relationships and identify critical links in many-to-many clusters"
)

with st.expander("How to use this tool", expanded=False):
    st.markdown(
        """
        **What this tool does**
        - Reads a mapping between two taxonomies/entities and classifies relationships.
        - Highlights many-to-many clusters and identifies critical links.
        - Proposes link-removal solutions that break many-to-many structures.

        **Input format required**
        - Upload a **CSV file with exactly 2 columns**.
        - Each row represents one mapping pair: `left_entity, right_entity`.
        - The first column is treated as the left side, the second as the right side.
        - Include a header row (recommended).
        - Example:

        ```csv
        IFRS_9,ESRS_E1
        IFRS_15,ESRS_E2
        IFRS_15,ESRS_E3
        ```

        **How to read the results**
        - `relation` column labels each row as one-to-one, many-to-one, one-to-many, or many-to-many.
        - `criticality_type = independent_critical` means removing that single link breaks the M-M pattern.
        - `critical_combinations` lists sets of links that must be removed together.
        - Each M-M component has a dedicated section with solutions and a small graph view.

        **Parent-Children detection (additional functionality)**
        - toggling the buttons below the displayed dataframe activates the functionality for the specified col.
        - logic: within each group of links linked to one correspondent for each component, recursive algorithm finds children of parents.
        - ex: if `ESRS_14_a` and `ESRS_14_a_i` are linked to the same entry on the other side, the latter gets highlighted as problematic.
        """
    )

SAMPLE_CSV_CONTENT = """left_entity,right_entity
IFRS_9,ESRS_E1
IFRS_15,ESRS_E2
IFRS_15,ESRS_E3
IFRS_5, ESRS_E4
IFRS_6, ESRS_E4
IFRS_7, ESRS_E4
IFRS_1, ESRS_c1
IFRS_1, ESRS_c2
IFRS_2, ESRS_c2
IFRS_3, ESRS_c3
IFRS_3, ESRS_c4
IFRS_4, ESRS_c4
IFRS_4, ESRS_c5
IFRS_5, ESRS_c5
IFRS_10, ESRS_c6
IFRS_10, ESRS_c7
IFRS_11, ESRS_c7
IFRS_11, ESRS_c8
IFRS_12, ESRS_c8
IFRS_12, ESRS_c9
IFRS_cycle1, ESRS_cycle1
IFRS_cycle1, ESRS_cycle2
IFRS_cycle2, ESRS_cycle1
IFRS_cycle2, ESRS_cycle2
"""

st.download_button(
    "Download sample CSV",
    data=SAMPLE_CSV_CONTENT,
    file_name="sample_mapping.csv",
    mime="text/csv",
    help="Use this sample to test the app format quickly.",
)


RELATION_COLORS = {
    "one-to-one": "",
    "many-to-one": "background-color: #d9f2d9",
    "one-to-many": "background-color: #ffe8cc",
    "many-to-many": "background-color: #f8d7da",
}


def style_relation(value: object) -> str | None:
    if not isinstance(value, str):
        return None

    return RELATION_COLORS.get(value)


def style_criticality_type(value: object) -> str | None:
    if value == "independent_critical":
        return "background-color: #c02716"
    # elif value == "non_critical_bridge":
    #     return "background-color: #ffc107"
    elif value == "cycle":
        return "background-color: #6c757d"

    return None


def format_critical_combinations_cell(value: object) -> list[list[str]]:
    """Render nested edge combinations while preserving list-style dataframe display."""
    if not isinstance(value, list) or len(value) == 0:
        return []

    formatted_sets: list[list[str]] = []
    for combo in value:
        if not isinstance(combo, (list, tuple)) or len(combo) == 0:
            continue

        edges_in_set: list[str] = []
        for edge in combo:
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                edges_in_set.append(f"{edge[0]} ↔ {edge[1]}")

        if edges_in_set:
            # Streamlit may render list items joined by ',' without spaces.
            # Prefix items after the first to keep a visual space after each comma.
            pretty_edges = [
                edge_text if idx == 0 else f" {edge_text}"
                for idx, edge_text in enumerate(edges_in_set)
            ]
            formatted_sets.append(pretty_edges)

    return formatted_sets


def create_component_visualization(component_graph, cols: Cols) -> str:
    """Create a compact two-column bipartite SVG visualization."""
    col1, col2 = cols.col_names

    # Separate nodes by type
    left_nodes = []
    right_nodes = []
    for node in component_graph.nodes():
        if col1 in node:
            left_nodes.append((node, node.replace(f"{col1}_", "")))
        else:
            right_nodes.append((node, node.replace(f"{col2}_", "")))

    # Layout parameters
    node_radius = 12
    v_spacing = 50
    vertical_padding = 24
    max_count = max(len(left_nodes), len(right_nodes), 1)
    page_height = (max_count - 1) * v_spacing + (vertical_padding * 2)

    # Calculate width dynamically based on longest label
    # Estimate: ~6.5 pixels per character at font-size 11, plus padding for text anchor positioning
    max_label_length = max(
        (max((len(label) for _, label in left_nodes), default=0)),
        (max((len(label) for _, label in right_nodes), default=0)),
    )
    char_width = 6.5
    label_pixel_width = max(
        max_label_length * char_width + 30, 100
    )  # +30 for node radius and spacing
    page_width = label_pixel_width * 2 + 100  # Both sides + center gap for edges

    # Calculate vertical positions (center shorter column)
    left_span = (max(len(left_nodes), 1) - 1) * v_spacing
    right_span = (max(len(right_nodes), 1) - 1) * v_spacing
    left_y_start = (page_height - left_span) / 2
    right_y_start = (page_height - right_span) / 2

    # Position nodes with adequate margin for label text
    left_node_x = label_pixel_width - 20
    right_node_x = page_width - label_pixel_width + 20

    # Map nodes to positions
    left_pos = {
        node: (left_node_x, left_y_start + i * v_spacing)
        for i, (node, _) in enumerate(left_nodes)
    }
    right_pos = {
        node: (right_node_x, right_y_start + i * v_spacing)
        for i, (node, _) in enumerate(right_nodes)
    }

    # Start SVG with overflow visible to allow text rendering beyond bounds
    svg = f'<svg width="{page_width}" height="{page_height}" xmlns="http://www.w3.org/2000/svg" overflow="visible" style="border: 1px solid #ddd; border-radius: 4px;">'

    # Draw edges first (so they appear behind nodes)
    for u, v in component_graph.edges():
        is_critical = (
            component_graph.edges[u, v].get("criticality_type")
            == "independent_critical"
        )
        edge_color = "#dc3545" if is_critical else "#999999"
        edge_width = 2 if is_critical else 1

        pos_u = left_pos.get(u) if u in left_pos else right_pos.get(u)
        pos_v = left_pos.get(v) if v in left_pos else right_pos.get(v)
        if pos_u is None or pos_v is None:
            continue

        x1 = pos_u[0]
        y1 = pos_u[1]
        x2 = pos_v[0]
        y2 = pos_v[1]

        svg += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{edge_color}" stroke-width="{edge_width}" />'

    # Draw left column nodes
    for node, label in left_nodes:
        x, y = left_pos[node]
        svg += f'<circle cx="{x}" cy="{y}" r="{node_radius}" fill="#7dd3fc" stroke="#0369a1" stroke-width="1.5" />'
        # Don't truncate anymore since width is dynamic
        safe_text = escape(label)
        svg += f'<text x="{x - node_radius - 8}" y="{y}" font-size="11" fill="#000" text-anchor="end" dominant-baseline="central">{safe_text}</text>'

    # Draw right column nodes
    for node, label in right_nodes:
        x, y = right_pos[node]
        svg += f'<circle cx="{x}" cy="{y}" r="{node_radius}" fill="#86efac" stroke="#16a34a" stroke-width="1.5" />'
        # Don't truncate anymore since width is dynamic
        safe_text = escape(label)
        svg += f'<text x="{x + node_radius + 8}" y="{y}" font-size="11" fill="#000" dominant-baseline="central">{safe_text}</text>'

    svg += "</svg>"
    return svg


uploaded_file = st.file_uploader("Upload mapping", type=["csv"])

if "children_issues_leftcol" not in st.session_state:
    st.session_state["children_issues_leftcol"] = False
if "children_issues_rightcol" not in st.session_state:
    st.session_state["children_issues_rightcol"] = False

if uploaded_file is not None:
    try:
        mapping_df = pd.read_csv(uploaded_file, sep=None)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
    else:
        if mapping_df.shape[1] != 2:
            st.error("The uploaded CSV must contain exactly 2 columns.")
        else:
            st.success("Mapping uploaded successfully.")

            children_issues_leftcol = st.session_state["children_issues_leftcol"]
            children_issues_rightcol = st.session_state["children_issues_rightcol"]

            mapped_df, graph, component_map = main(
                mapping_df,
                children_issues_leftcol=children_issues_leftcol,
                children_issues_rightcol=children_issues_rightcol,
            )
            cols = Cols(mapped_df)
            col1, col2 = cols.col_names

            display_df = mapped_df.copy()
            display_df["__row_id__"] = range(len(display_df))
            if "critical_combinations" in display_df.columns:
                display_df["critical_combinations"] = display_df[
                    "critical_combinations"
                ].apply(format_critical_combinations_cell)

            st.subheader("Mapping Results")

            # Prepare display dataframe without boolean columns but keep them in mapped_df for styling
            problematic_col1 = f"is_problematic_child_{col1}"
            problematic_col2 = f"is_problematic_child_{col2}"

            display_cols = [
                c
                for c in display_df.columns
                if not c.startswith("is_problematic_child_") and c != "__row_id__"
            ]

            # Sort by component_id to group rows from the same component together
            if "component_id" in display_df.columns:
                display_df = display_df.sort_values("component_id", kind="stable")
                display_df = display_df.reset_index(drop=True)

            # Keep stable source row mapping after optional sorting/reset.
            display_row_id_by_index = display_df["__row_id__"].copy()
            display_df = display_df[display_cols]

            # Apply styling
            styled_df = display_df.style.map(style_relation, subset=["relation"])
            styled_df = styled_df.map(
                style_criticality_type, subset=["criticality_type"]
            )

            # Apply children issues styling using the original mapped_df data
            if children_issues_leftcol and problematic_col1 in mapped_df.columns:

                def style_col1_children(s):
                    return [
                        "background-color: #eabd8a"
                        if mapped_df.iloc[int(display_row_id_by_index.loc[i])][
                            problematic_col1
                        ]
                        else ""
                        for i in s.index
                    ]

                styled_df = styled_df.apply(style_col1_children, subset=[col1])

            if children_issues_rightcol and problematic_col2 in mapped_df.columns:

                def style_col2_children(s):
                    return [
                        "background-color: #eabd8a"
                        if mapped_df.iloc[int(display_row_id_by_index.loc[i])][
                            problematic_col2
                        ]
                        else ""
                        for i in s.index
                    ]

                styled_df = styled_df.apply(style_col2_children, subset=[col2])

            st.dataframe(styled_df, use_container_width=True)

            _, toggle = st.columns([3, 1])

            with toggle:
                st.caption("Enable child-parent issue detection for column of choice")

                st.checkbox(
                    f"'{col1}' column",
                    key="children_issues_leftcol",
                )

                st.checkbox(
                    f"'{col2}' column",
                    key="children_issues_rightcol",
                )

            # Display solutions organized by many-to-many component
            m2m_component_ids = [
                comp_id
                for comp_id, info in component_map.items()
                if info.get("is_many_to_many")
            ]

            if m2m_component_ids:
                for comp_id in sorted(m2m_component_ids):
                    comp_info = component_map[comp_id]

                    # Create component section
                    st.markdown(f"#### M-M Component ID {comp_id}")

                    # Extract critical edges and combinations for this component
                    component_critical_edges = []
                    component_combinations = {}

                    for u, v in graph.edges():
                        edge_comp_id = graph.edges[u, v].get("component_id")
                        if edge_comp_id != comp_id:
                            continue

                        if (
                            graph.edges[u, v].get("criticality_type")
                            == "independent_critical"
                        ):
                            component_critical_edges.append((u, v))

                        combos = graph.edges[u, v].get("critical_combinations", [])
                        if combos:
                            for combo in combos:
                                combo_key = tuple(sorted([str(e) for e in combo]))
                                if combo_key not in component_combinations:
                                    component_combinations[combo_key] = combo

                    # Display solutions for this component
                    if len(component_critical_edges) > 0:
                        st.info("**📌 Independently Critical Links**")
                        st.write(
                            "Removing any **one** of these links will break the many-to-many relationship:"
                        )
                        for u, v in component_critical_edges:
                            st.markdown(f"• {u} ↔ {v}")

                    elif len(component_combinations) > 0:
                        st.info("**🔗 Multi-Link Solutions**")
                        st.write(
                            "Remove **all links in a group** to break the many-to-many relationship:"
                        )
                        for combo_idx, (combo_key, edges) in enumerate(
                            list(component_combinations.items())[:10], 1
                        ):
                            with st.expander(
                                f"Solution {combo_idx}: Remove {len(edges)} links"
                            ):
                                st.markdown("**Links to remove together:**")
                                for edge in edges:
                                    st.markdown(f"• {edge[0]} ↔ {edge[1]}")
                                st.info(
                                    f"Removing all {len(edges)} of these links breaks the M-N structure."
                                )
                        if len(component_combinations) > 10:
                            st.info(
                                f"ℹ️ {len(component_combinations) - 10} additional solutions exist (showing top 10)"
                            )
                    else:
                        st.warning(
                            "⚠️ **Complex Cyclic Cluster**: No simple solutions exist. Manual intervention may be needed."
                        )

                    # Display network visualization for this component
                    component_nodes = component_map[comp_id]["nodes"]
                    component_subgraph = nx.subgraph(graph, component_nodes)

                    svg_html = create_component_visualization(component_subgraph, cols)
                    st.markdown(
                        (
                            '<div style="display:flex;justify-content:center;">'
                            f"{svg_html}"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )

                    st.divider()
            else:
                st.info("No many-to-many components found in the mapping.")

            original_name = uploaded_file.name or "mapping.csv"
            base_name = original_name.rsplit(".", 1)[0]
            output_name = f"{base_name}_processed.csv"

            st.download_button(
                "Download full processed data",
                data=mapped_df.to_csv(index=False).encode("utf-8"),
                file_name=output_name,
                mime="text/csv",
                help="Download the full processed dataframe as CSV.",
            )
