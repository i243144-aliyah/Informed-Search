import streamlit as st
import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt

# Graph, Use Case: Emergency Supply Robot
locations = {
    "Pharmacy": (0, 0),
    "Main_Corridor": (2, 1),
    "Patient_Wing": (1, 4),
    "Nursing_Station": (4, 2),
    "Laboratory": (5, 5),
    "Emergency_Ward": (8, 6)
}

hospital_graph = {
    "Pharmacy": {
        "Main_Corridor": 2.2,
        "Patient_Wing": 4.1
    },
    "Main_Corridor": {
        "Nursing_Station": 2.2
    },
    "Patient_Wing": {
        "Laboratory": 5.0
    },
    "Nursing_Station": {
        "Laboratory": 3.2,
        "Emergency_Ward": 6.0
    },
    "Laboratory": {
        "Emergency_Ward": 3.2
    },
    "Emergency_Ward": {}
}

# Heuristic: Euclidean Distance
def heuristic(current, goal):
    x1, y1 = locations[current]
    x2, y2 = locations[goal]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Path reconstruction
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# GBFS (Greedy Best-First Search)
def gbfs(start, goal):
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start))
    came_from = {}
    g_score = {node: float('inf') for node in hospital_graph}
    g_score[start] = 0
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current), g_score[goal]

        if current in visited:
            continue
        visited.add(current)

        for neighbor, weight in hospital_graph[current].items():
            if neighbor not in visited:
                if g_score[current] + weight < g_score[neighbor]:
                    g_score[neighbor] = g_score[current] + weight
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (heuristic(neighbor, goal), neighbor))

    return None, float('inf')

# A* Search Algorithm
def a_star(start, goal):
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start))
    came_from = {}
    g_score = {node: float('inf') for node in hospital_graph}
    g_score[start] = 0

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current), g_score[goal]

        for neighbor, weight in hospital_graph[current].items():
            tentative_g_score = g_score[current] + weight
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    return None, float('inf')

##########################################
# Streamlit GUI Code

# Set Page Config
st.set_page_config(page_title="Emergency Robot Pathfinding", layout="centered")

# Title and Description
st.title("🏥 Emergency Supply Robot Pathfinding")
st.write("Find optimal paths for emergency supply delivery in a hospital graph using **GBFS** or **A*** algorithm.")

# Define nodes list
nodes = list(hospital_graph.keys())

# Selectboxes for user inputs
start = st.selectbox(
    "Select Initial Node",
    options=nodes,
    index=nodes.index("Pharmacy")
)

goal = st.selectbox(
    "Select Goal Node",
    options=nodes,
    index=nodes.index("Emergency_Ward")
)

algorithm = st.selectbox(
    "Select Search Algorithm",
    options=["A*", "GBFS"]
)

# Run Search Button Action
if st.button("Run Search"):
    if algorithm == "GBFS":
        path, cost = gbfs(start, goal)
    else:
        path, cost = a_star(start, goal)

    if path is None or cost == float('inf'):
        st.error(f"No valid path found from {start} to {goal}.")
    else:
        # Display numerical results
        st.subheader("Search Result")
        st.write(f"**Algorithm:** {algorithm}")
        st.write(f"**Solution Path:** {' → '.join(path)}")
        st.write(f"**Total Path Cost:** {cost:.2f}")

        # Visualize NetworkX graph
        G = nx.DiGraph()
        for node, neighbors in hospital_graph.items():
            for neighbor, weight in neighbors.items():
                G.add_edge(node, neighbor, weight=weight)

        pos = locations
        fig, ax = plt.subplots(figsize=(10, 6))

        # Build list of path edges to highlight
        path_edges = list(zip(path[:-1], path[1:]))

        # Draw default background graph
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1200, node_color="skyblue")
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", arrows=True, arrowsize=15, width=1.5)

        # Highlight path nodes and edges
        nx.draw_networkx_nodes(G, pos, nodelist=path, ax=ax, node_color="lightgreen", node_size=1300)
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=ax, edge_color="red", width=3, arrows=True, arrowsize=20)

        # Draw edge weights
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8)

        ax.set_title(f"{algorithm} Solution Path ({start} to {goal})")
        ax.axis("off")

        st.pyplot(fig)