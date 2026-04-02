"""
Test suite for critical edge detection in many-to-many mapping clusters.

This test suite validates:
1. Independent critical edges - edges whose single removal breaks M-N
2. Critical edge combinations - sets of edges whose joint removal breaks M-N
3. Complex clusters - cyclic structures that require multi-edge fixes
4. Position classification - identifying dead-end vs interior edges
"""

from typing import Tuple, List
import pandas as pd
import networkx as nx
import pytest
from analyse import (
    Cols,
    main,
    create_relationships,
    create_graph_from_df,
    is_many_to_many_component,
    update_graph_with_attributes,
    apply_edge_attribute_to_df,
    apply_mapping_to_df,
    get_edgesCategories_map_from_attribute,
    return_critical_edges_from_graph,
    find_critical_edge_combinations,
    analyse_childparents_issues,
)


def build_test_graph(data: dict, add_attributes: bool = True):
    """Shared setup used across graph-based tests."""
    df = pd.DataFrame(data)
    cols = Cols(df)

    create_relationships(df, cols)
    G = create_graph_from_df(df, cols)
    components = list(nx.connected_components(G))

    if add_attributes:
        update_graph_with_attributes(components, G, cols)

    return cols, G, components


def iter_and_return(
    graph: nx.Graph, attr: str, attr_cat: str | None = None, edges: bool = True
) -> List[Tuple[str, str]]:
    """
    Outil to iterate over edges or nodes based on attributes.
    Return a list of edges/nodes depending of what's specified.
    """
    if edges:
        if attr_cat is None:
            return [(u, v) for u, v in graph.edges() if graph.edges[u, v].get(attr)]
        else:
            return [
                (u, v)
                for u, v in graph.edges()
                if graph.edges[u, v].get(attr) == attr_cat
            ]
    else:
        if attr_cat is None:
            return [nod for nod in graph.nodes() if graph.nodes[nod].get(attr)]
        else:
            return [
                nod for nod in graph.nodes() if graph.nodes[nod].get(attr) == attr_cat
            ]


class TestSimple:
    """Test case: simple pattern with 5 nodes and 3 edges. No many-to-many"""

    @pytest.fixture
    def setup(self):
        data = {"left": ["A", "C", "E"], "right": ["B", "D", "B"]}
        return build_test_graph(data, add_attributes=False)

    def test_is_many_to_many(self, setup):
        """Verify the component is indeed many-to-many"""
        cols, G, components = setup
        for comp in components:
            assert is_many_to_many_component(comp, G, cols) is False


class TestMM_3edges:
    """Test case: simple many_to_many component with 3 edges. All edges are critical links"""

    @pytest.fixture
    def setup(self):
        data = {"left": ["A", "C", "C"], "right": ["B", "B", "D"]}
        return build_test_graph(data)

    def test_is_many_to_many(self, setup):
        """Verify the component is many-to-many"""
        cols, G, components = setup

        for comp in components:
            assert is_many_to_many_component(comp, G, cols)

    def test_positioninpath_attr(self, setup):
        """Test attribute: positions in graph"""
        _, G, _ = setup

        posinpath_edges = iter_and_return(G, "position_in_path")

        assert posinpath_edges
        assert len(posinpath_edges) == 3

        dead_ends = iter_and_return(G, "position_in_path", "dead_end_bridge")
        interiors = iter_and_return(G, "position_in_path", "interior_bridge")

        assert len(dead_ends) == 2
        assert len(interiors) == 1

    def test_criticality_attr(self, setup):
        """Test if all edges are critical"""
        _, G, _ = setup

        critical_edges = return_critical_edges_from_graph(G)

        assert len(critical_edges) == 3  # all 3 edges should be critical


class TestMM_4edges:
    """Test case: simple many_to_many component with 4 edges. The edges in the middle are critical links"""

    @pytest.fixture
    def setup(self):
        data = {"left": ["A", "C", "C", "E"], "right": ["B", "B", "D", "D"]}
        return build_test_graph(data)

    def test_is_many_to_many(self, setup):
        """Verify the component is many-to-many"""
        cols, G, components = setup

        for comp in components:
            assert is_many_to_many_component(comp, G, cols)

    def test_positioninpath_attr(self, setup):
        """Test attribute: positions in graph"""
        _, G, _ = setup

        posinpath_edges = iter_and_return(G, "position_in_path")

        assert posinpath_edges
        assert len(posinpath_edges) == 4

        dead_ends = iter_and_return(G, "position_in_path", "dead_end_bridge")
        interiors = iter_and_return(G, "position_in_path", "interior_bridge")

        assert len(dead_ends) == 2
        assert len(interiors) == 2

    def test_criticality_attr(self, setup):
        """Test if all edges are critical"""
        _, G, _ = setup

        critical_edges = return_critical_edges_from_graph(G)
        criticals_inferred = [("right_B", "left_C"), ("left_C", "right_D")]

        for edge in critical_edges:
            assert edge in criticals_inferred or edge[::-1] in criticals_inferred


class TestMM_5edges:
    """Test case: simple many_to_many component with 5 edges. Critical link in the middle"""

    @pytest.fixture
    def setup(self):
        data = {"left": ["A", "C", "C", "E", "E"], "right": ["B", "B", "D", "D", "F"]}
        return build_test_graph(data)

    def test_positioninpath_attr(self, setup):
        """Test attribute: positions in graph"""
        _, G, _ = setup

        posinpath_edges = iter_and_return(G, "position_in_path")

        assert posinpath_edges
        assert len(posinpath_edges) == 5

        dead_ends = iter_and_return(G, "position_in_path", "dead_end_bridge")
        interiors = iter_and_return(G, "position_in_path", "interior_bridge")

        assert len(dead_ends) == 2
        assert len(interiors) == 3

    def test_criticality_attr(self, setup):
        """Test if all edges are critical"""
        _, G, _ = setup

        critical_edges = return_critical_edges_from_graph(G)

        # 1 central critical edge
        assert len(critical_edges) == 1
        assert critical_edges[0] == ("left_C", "right_D") or critical_edges[0] == (
            "right_D",
            "left_C",
        )


class TestMM_6edges:
    """Test case: many_to_many component with 6 edges. First case with combinations"""

    @pytest.fixture
    def setup(self):
        data = {
            "left": ["A", "C", "C", "E", "E", "G"],
            "right": ["B", "B", "D", "D", "F", "F"],
        }
        return build_test_graph(data)

    def test_positioninpath_attr(self, setup):
        """Test attribute: positions in graph"""
        _, G, _ = setup

        posinpath_edges = iter_and_return(G, "position_in_path")

        assert posinpath_edges
        assert len(posinpath_edges) == 6

        dead_ends = iter_and_return(G, "position_in_path", "dead_end_bridge")
        interiors = iter_and_return(G, "position_in_path", "interior_bridge")

        assert len(dead_ends) == 2
        assert len(interiors) == 4

    def test_criticality_attr(self, setup):
        """Test if all edges are critical"""
        _, G, _ = setup

        critical_edges = return_critical_edges_from_graph(G)
        assert not critical_edges

    def test_combos(self, setup):
        """Test if (only) the 6 potential combos are found"""
        cols, G, components = setup

        counter = 0
        combs = G.edges(data="critical_combinations")
        for edge in list(combs):
            counter += len(edge[2])

        # tot number of edge sets attributes added to edges
        assert counter == 12

        dict_comb = find_critical_edge_combinations(
            G, cols, list(G.edges), max_combination_size=3
        )
        assert len(dict_comb) == 6


class TestMM_7edges:
    @pytest.fixture
    def setup(self):
        data = {
            "left": ["A", "C", "C", "E", "E", "G", "G"],
            "right": ["B", "B", "D", "D", "F", "F", "H"],
        }
        return build_test_graph(data)

    def test_criticality_attr(self, setup):
        """Test if all edges are critical"""
        _, G, _ = setup

        critical_edges = return_critical_edges_from_graph(G)
        criticals_inferred = [("right_D", "left_C"), ("left_E", "right_F")]

        for edge in critical_edges:
            assert edge in criticals_inferred or edge[::-1] in criticals_inferred


class TestMM_8edges:
    @pytest.fixture
    def setup(self):
        data = {
            "left": ["A", "C", "C", "E", "E", "G", "G", "I"],
            "right": ["B", "B", "D", "D", "F", "F", "H", "H"],
        }
        return build_test_graph(data)

    def test_criticality_attr(self, setup):
        """Test if all edges are critical"""
        _, G, _ = setup

        critical_edges = return_critical_edges_from_graph(G)
        criticals_inferred = [("right_D", "left_C"), ("left_G", "right_F")]

        for edge in critical_edges:
            assert edge in criticals_inferred or edge[::-1] in criticals_inferred


class TestCyclicComponent:
    """Test case: Simple cycle (4 edges, 4 nodes)"""

    @pytest.fixture
    def setup(self):
        data = {"left": ["A", "A", "C", "C"], "right": ["B", "D", "B", "D"]}
        return build_test_graph(data)

    def test_cyclic_is_many_to_many(self, setup):
        """Verify cycle is many-to-many"""
        cols, G, components = setup

        for comp in components:
            assert is_many_to_many_component(comp, G, cols)

    def test_cyclic_flagged_as_complex(self, setup):
        """Test that cycles are detected as complex"""
        _, G, _ = setup

        critical_edges = return_critical_edges_from_graph(G)
        assert len(critical_edges) <= 1

        for edge in G.edges(data=True):
            assert edge[2].get("criticality_type") == "cycle"


class TestMultipleMMComponents:
    @pytest.fixture
    def setup(self):
        df = pd.read_csv("examples/ex_M-N_tanticomponents.csv", delimiter=";")
        esrs = df["ESRS"].tolist()
        ifrs = df["IFRS"].tolist()

        data = {"ESRS": esrs, "IFRS": ifrs}
        df = pd.DataFrame(data)
        cols = Cols(df)

        create_relationships(df, cols)
        G = create_graph_from_df(df, cols)
        components = list(nx.connected_components(G))
        component_map = update_graph_with_attributes(components, G, cols)
        apply_edge_attribute_to_df(df, cols, "critical_combinations", G)
        apply_mapping_to_df(
            df,
            cols,
            "criticality_type",
            get_edgesCategories_map_from_attribute(G, "criticality_type"),
        )

        return df, cols, G, components, component_map

    def test_combinations(self, setup):
        _, _, G, _, _ = setup

        combinations = list(G.edges(data="critical_combinations"))
        for edge in combinations:
            if edge[2] is None:
                continue
            assert isinstance(edge[2], (list, tuple))

    def test_df_critical_combinations_or_critical_edges_exist(self, setup):
        df, _, _, _, _ = setup

        assert "critical_combinations" in df.columns
        assert "criticality_type" in df.columns

        has_combo_rows = df["critical_combinations"].notna().any()
        has_independent_critical = (
            df["criticality_type"] == "independent_critical"
        ).any()

        assert has_combo_rows or has_independent_critical

    def test_component_map_not_all_complex(self, setup):
        _, _, _, _, component_map = setup

        m2m_components = [
            info for info in component_map.values() if info.get("is_many_to_many")
        ]
        m2m_states = {info.get("criticality_type") for info in m2m_components}

        assert m2m_states.intersection({"has_critical_edges", "has_critical_combos"})
        assert not m2m_states.issubset({"complex_cyclic", "cycle"})


class TestDisconnectedComponentsIsolation:
    @pytest.fixture
    def setup(self):
        data = {
            "left": ["A", "C", "C", "E", "E", "G", "X", "X", "Z", "Z"],
            "right": ["B", "B", "D", "D", "F", "F", "Y", "W", "Y", "W"],
        }
        cols, G, components = build_test_graph(data)
        component_map = update_graph_with_attributes(components, G, cols)

        return cols, G, component_map

    def test_combo_component_not_masked_by_other_component(self, setup):
        _, G, component_map = setup

        # Ensure at least one component gets combination solutions.
        states = [
            info.get("criticality_type")
            for info in component_map.values()
            if info.get("is_many_to_many")
        ]
        assert "has_critical_combos" in states

        # Ensure combo attributes are present on the graph edges.
        combo_edges = [
            (u, v)
            for u, v in G.edges()
            if isinstance(G.edges[u, v].get("critical_combinations"), list)
            and len(G.edges[u, v].get("critical_combinations", [])) > 0
        ]
        assert combo_edges


class TestAlgorithmParentChildren:
    def test_simplechildren(self):
        series = pd.Series(
            [
                "ESRS2_GDR-A_45_a_1",
                "ESRS2_GDR-A_45_a_2",
                "ESRS2_GDR-A_45_a_3",
                "ESRS2_GDR-A_45_b",
                "ESRS2_GDR-A_46_a_1",
                "ESRS2_GDR-A_46_a_2",
                "ESRS2_GDR-A_46_b",
                "ESRS2_GDR-A_46_c",
                "E1_E1-5_20",
            ]
        )
        analyse_childparents_issues(series, probl_ids := set())

        assert len(probl_ids) == 0

    def test_duplicate_index_maps_problematic_rows_correctly(self):
        df = pd.DataFrame(
            {
                "left": ["A_1", "A_1_1", "B_1"],
                "right": ["X", "X", "X"],
            },
            index=[5, 5, 7],
        )

        analysed_df, _, _ = main(
            df,
            children_issues_leftcol=True,
            children_issues_rightcol=False,
        )

        assert analysed_df["is_problematic_child_left"].tolist() == [False, True, False]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
