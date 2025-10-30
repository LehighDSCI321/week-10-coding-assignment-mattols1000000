"""
This module contains a class for VersatileDigraph(),
and the BinaryGraph() subclass of the VersatileDigraph() class
"""
from collections import deque
import importlib
class VersatileDigraph():
    """VersitileDigraph() can be used to represent a network,
    a series of nodes and the edges that connect them"""
    def __init__(self):
        """This is the contstructor method for the class.
        We make self.nodes and self.edges as dictionaries for internal use"""
        self.nodes = {} # initializing nodes dict
        self.edges = {} # initializing edges dict

    def add_edge(self, start_node_id, end_node_id, start_node_value = 0, end_node_value = 0,
                 edge_name = None, edge_weight = None):
        """This method can add a representation of an edge
        (details of starting and ending nodes, edge weight and name) into the object when called"""
        if edge_weight is not None and not isinstance(edge_weight, (int, float)):
            raise TypeError(f"The given edge_weight '{edge_weight}' must be an int or float")
        if edge_weight is not None and edge_weight < 0:
            raise ValueError(f"The given edge_weight '{edge_weight}' must be non-negative")
        self.add_node(start_node_id, start_node_value)
        self.add_node(end_node_id, end_node_value)
        if start_node_id not in self.edges:
            self.edges[start_node_id] = {}
        if edge_name is None:
            edge_name = str(start_node_id) + str(end_node_id)
        if edge_name in self.edges[start_node_id]:
            raise ValueError(f"Edge name '{edge_name}' from '{start_node_id}' already exists.")
        self.edges[start_node_id][edge_name] = (edge_weight, end_node_id)

    def add_node(self, node_id, node_value = 0):
        """This method can add nodes to the network"""
        if not isinstance(node_value, (int, float)):
            raise TypeError(f"node_value '{node_value}' must be an int or float")
        self.nodes.update({node_id : node_value})

    def get_nodes(self):
        """This method returns a list of the nodes in the graph"""
        return list(self.nodes.keys()) # returns the nodes in the graph

    def get_edge_weight(self, start_node_id, end_node_id):
        """This method returns an edge weight for a specific starting and ending nodes"""
        for edge_name in list(self.edges[start_node_id].keys()):
            if self.edges[start_node_id][edge_name][1] == end_node_id:
                return self.edges[start_node_id][edge_name][0]
        raise KeyError(f"An edge from '{start_node_id}' to '{end_node_id}' does not exist")

    def get_node_value(self, node_id):
        """This method returns values of nodes"""
        if node_id not in self.get_nodes():
            raise KeyError(f"The node, '{node_id}' does not exist in the graph")
        return self.nodes[node_id]

    def in_degree(self, node_id):
        """This method returns the indegree of a given node (node_id)"""
        if node_id not in self.get_nodes():
            raise KeyError(f"The node, '{node_id}' does not exist in the graph")
        return sum(
            1
            for start_node_id, edge_dict in self.edges.items()
            for edge_name, (edge_weight, end_node_id) in edge_dict.items()
            if end_node_id == node_id
            )

    def out_degree(self, node_id):
        """This method returns the outdegree of a given node (node_id)"""
        if node_id not in self.get_nodes():
            raise KeyError(f"The node, '{node_id}' does not exist in the graph")
        return sum(
            1
            for edge_name, (edge_weight, end_node_id) in self.edges.get(node_id, {}).items()
            )

    def predecessors(self, node_id):
        """This method returns the predecessor(s) of a given node (node_id)"""
        if node_id not in self.get_nodes():
            raise KeyError(f"The node, '{node_id}' does not exist in the graph")
        return [
            start_node_id
            for start_node_id, edge_dict in self.edges.items()
            for edge_name, (edge_weight, end_node_id) in edge_dict.items()
            if end_node_id == node_id
            ]

    def successors(self, node_id):
        """This method returns the successor(s) of a given node (node_id)"""
        if node_id not in self.get_nodes():
            raise KeyError(f"The node, '{node_id}' does not exist in the graph")
        return [end_node_id for eg_nm, (eg_wt, end_node_id) in self.edges.get(node_id, {}).items()]

    def successor_on_edge(self, node_id, edge_name):
        """This method returns the successor of an edge (edge_name) from a given node (node_id)"""
        if node_id not in self.get_nodes() or edge_name not in self.edges:
            raise KeyError(f"The node, '{node_id}' or the edge '{edge_name}' is not in the graph")
        return (
            self.edges[node_id][edge_name][1]
            if node_id in self.edges and edge_name in self.edges[node_id]
            else []
        )

    def plot_graph(self):
        '''This method creates a visual representation of the digraph using graphviz'''
        graphviz = importlib.import_module("graphviz")
        gr = graphviz.Digraph()
        list_of_succ_lists = []
        list_of_lonely_nodes = []
        for node_id in self.get_nodes():
            succs = [end_node_id
            for edge_name, (edge_weight, end_node_id) in self.edges.get(node_id, {}).items()]
            list_of_succ_lists.append(succs)
            if succs == []:
                list_of_lonely_nodes.append(node_id)
        for node_id, succs in zip(self.get_nodes(), list_of_succ_lists):
            for succ in succs:
                gr.edge(node_id, succ)
        for node_id in list_of_lonely_nodes:
            gr.node(node_id)
        return gr.view()

    def plot_edge_weights(self):
        '''This method plots edge weights by edge names on a histogram using Bokeh'''
        bokehplot = importlib.import_module("bokeh.plotting")
        bokehio = importlib.import_module("bokeh.io")
        figure = getattr(bokehplot, "figure") # importing figure() from bokeh.plotting
        show = getattr(bokehio, "show") # importing show() from bokeh.io
        edge_name_nest = []
        edge_weight_list = []
        for start_node_id in list(self.edges.keys()):
            edge_name_nest.append(list(self.edges[start_node_id].keys()))
            for edge_name, (edge_weight, _) in self.edges[start_node_id].items():
                edge_weight_list.append(edge_weight)
                if edge_weight is None:
                    raise ValueError("edge_weight must be given to all edges to plot_edge_weights")
        edge_nm_list = list({edge_name for sublst in edge_name_nest for edge_name in sublst})
        hist = figure(
            x_range = edge_nm_list,
            title="Edge Weights by Edge Names",
            toolbar_location=None, tools=""
        )
        hist.vbar(x = edge_nm_list, top = edge_weight_list, width=0.7)
        hist.xgrid.grid_line_color = None
        hist.y_range.start = 0
        return show(hist)

    def print_graph(self):
        """This method prints sentence representations of the relationships in the entire graph"""
        for node_id, value in self.nodes.items():
            print(f"Node {node_id} with value {value}")
        for start_node_id, edge_dict in self.edges.items():
            for edge_name, (edge_weight, end_node_id) in edge_dict.items():
                print((
                    f"Edge from {start_node_id} to {end_node_id} "
                    f"with weight {edge_weight} and name {edge_name}"
                ))

class BinaryGraph(VersatileDigraph):
    """This is a subclass of the VersatileDigraph class that creates a
    binary tree through inheriting attributes and methods from the VersatileDigraph class"""
    def __init__(self):
        """This is the constructor method for the class. Adds a "Root" parent node by default"""
        super().__init__()
        self.add_node("Root")

    def add_node_left(self, child_id, child_value, parent_id = "Root"):
        """This method adds a new left child under an existing parent"""
        if parent_id not in self.nodes:
            super().add_node(parent_id)
        if child_id in self.nodes:
            raise ValueError(f"Child '{child_id}' already exists; expected a new node.")
        if "left" in self.edges.get(parent_id, {}):
            raise ValueError(f"Parent '{parent_id}' already has a left child.")

        self.add_node(child_id, child_value)
        super().add_edge(
            parent_id, child_id,
            start_node_value=self.get_node_value(parent_id),
            end_node_value=child_value,
            edge_name="left",
            edge_weight=None
        )

    def add_node_right(self, child_id, child_value, parent_id = "Root"):
        """This method adds a new right child under an existing parent"""
        if parent_id not in self.nodes:
            super().add_node(parent_id)
        if child_id in self.nodes:
            raise ValueError(f"Child '{child_id}' already exists; expected a new node.")
        if "right" in self.edges.get(parent_id, {}):
            raise ValueError(f"Parent '{parent_id}' already has a right child.")

        self.add_node(child_id, child_value)
        super().add_edge(
            parent_id, child_id,
            start_node_value=self.get_node_value(parent_id),
            end_node_value=child_value,
            edge_name="right",
            edge_weight=None
        )

    def get_node_left(self, parent_id):
        """This method gets the left childs ID given the parent node id, None if no left child"""
        if parent_id not in self.nodes:
            raise KeyError(f"Parent node '{parent_id}' does not exist.")
        try:
            return self.edges[parent_id]["left"][1]
        except KeyError:
            return None

    def get_node_right(self, parent_id):
        """This method gets the right childs ID given the parent node id, None if no right child"""
        if parent_id not in self.nodes:
            raise KeyError(f"Parent node '{parent_id}' does not exist.")
        try:
            return self.edges[parent_id]["right"][1]
        except KeyError:
            return None

class SortingTree(BinaryGraph):
    """This is a subclass of the BinaryGraph class that creates a sorting tree"""
    def __init__(self, root_value = None):
        """This is the constructor method for the class. Has an optional root_value kwarg"""
        super().__init__()
        if root_value is not None:
            self.edges.clear()
            self.nodes["Root"] = root_value
    def insert(self, node_value):
        """This method correctly inserts a node into the sorting tree 
        through inheriting attributes and methods from the BinaryGraph class"""
        if not isinstance(node_value, (int, float)):
            raise TypeError(f"value '{node_value}' must be numeric")
        def travel(node_id):
            now = self.get_node_value(node_id)
            if node_value < now:
                if self.get_node_left(node_id) is None:
                    child_id = f"n{len(self.nodes)+1}"
                    self.add_node_left(child_id, node_value, parent_id = node_id)
                    return
                return travel(self.get_node_left(node_id))
            else:
                if self.get_node_right(node_id) is None:
                    child_id = f"n{len(self.nodes)+1}"
                    self.add_node_right(child_id, node_value, parent_id = node_id)
                    return
                return travel(self.get_node_right(node_id))
        travel("Root")
    def traverse(self):
        """This method returns a list of sorted node values"""
        if (
            "Root" in self.nodes
            and self.get_node_left("Root") is None
            and self.get_node_right("Root") is None
            and len(self.nodes) == 1
        ):
            print()
            return [self.get_node_value("Root")]

        out = []

        def _order(node_id):
            if node_id is None:
                return
            left = self.get_node_left(node_id)
            if left is not None:
                _order(left)
            out.append(self.get_node_value(node_id))
            right = self.get_node_right(node_id)
            if right is not None:
                _order(right)

        _order("Root")
        for item in out:
            print(item, end=" ")

class SortableDigraph(VersatileDigraph):
    """Subclass of VersatileDigraph that can topologically sort nodes."""
    def __init__(self):
        """Constructor method for the class"""
        super().__init__()
        self._last_toposort = None

    def top_sort(self):
        """Topologically sort nodes modeled after Python Algorithms 4-10"""
        nodes = list(self.get_nodes())
        count = {u: 0 for u in nodes}
        for u in nodes:
            for _, (_, v) in self.edges.get(u, {}).items():
                count[v] = count.get(v, 0) + 1
        que = [u for u in nodes if count[u] == 0]
        sort = []
        while que:
            u = que.pop()
            sort.append(u)
            for _, (_, v) in self.edges.get(u, {}).items():
                count[v] -= 1
                if count[v] == 0:
                    que.append(v)
        if len(sort) != len(nodes):
            raise ValueError("Graph is not sortable due to cycle(s)")
        return sort

    def _has_cycle(self):
        """Return True iff the graph contains a cycle (Kahn's algorithm-based)."""
        nodes = self.get_nodes()
        indeg = {u: 0 for u in nodes}
        for u, edge_dict in self.edges.items():
            for _, (_, v) in edge_dict.items():
                if v not in indeg:
                    indeg[v] = 0
                indeg[v] += 1

        q = deque([u for u in nodes if indeg.get(u, 0) == 0])
        seen = 0
        while q:
            u = q.popleft()
            seen += 1
            for _, (_, v) in self.edges.get(u, {}).items():
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return seen != len(nodes)

class TraversableDigraph(SortableDigraph):
    """This class contains methods for Depth-first Search and Breadth-first Search"""
    def dfs(self, start_node):
        """Iterative DFS generator starting at start_node (like iter_dfs)."""
        if start_node not in self.nodes:
            raise KeyError(f"Start node '{start_node}' does not exist.")
        S, Q = set(), []
        Q.append(start_node)
        while Q:
            u = Q.pop()
            if u in S:
                continue
            S.add(u)
            yield u
            Q.extend(self.successors(u))
    def bfs(self, start_node, qtype = None):
        """This method performs a breadth-first search traversal of the digraph"""
        if start_node not in self.nodes:
            raise KeyError(f"Start node '{start_node}' does not exist.")
        if qtype is None:
            class _fifo(deque):
                add = deque.append
                def pop(self):
                    return deque.popleft(self)
            Q = _fifo()
        else:
            Q = qtype()
        S = set()
        Q.add(start_node)
        while Q:
            u = Q.pop()
            if u in S:
                continue
            S.add(u)
            for v in self.successors(u):
                Q.add(v)
            yield u

class DAG(TraversableDigraph):
    """Keep the graph acyclic by rolling back any edge that would create a cycle."""
    def add_edge(self, start_node_id, end_node_id,
                 start_node_value=0, end_node_value=0,
                 edge_name=None, edge_weight=None):
        super().add_edge(start_node_id, end_node_id,
                         start_node_value, end_node_value,
                         edge_name, edge_weight)
        used_name = edge_name if edge_name is not None else f"{start_node_id}{end_node_id}"
        if self._has_cycle():
            try:
                del self.edges[start_node_id][used_name]
                if not self.edges[start_node_id]:
                    del self.edges[start_node_id]
            except KeyError:
                pass
            raise ValueError("The added edge introduces a cycle")
        return None
