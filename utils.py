import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def generate_cus_grains(box_size, target_vol_fraction, mean_r, sigma_r):
    # box_size = (L, L, T) where T is film thickness
    L_x, L_y, T = box_size
    spheres = []
    current_volume = 0
    box_volume = L_x * L_y * T
    
    while (current_volume / box_volume) < target_vol_fraction:
        # Sample radius from a realistic Log-Normal distribution
        r = np.random.lognormal(mean=np.log(mean_r), sigma=sigma_r)
        
        # Random coordinates inside the thin film
        x = np.random.uniform(0, L_x)
        y = np.random.uniform(0, L_y)
        z = np.random.uniform(0, T)
        
        spheres.append((x, y, z, r))
        
        # Note: True overlapping volume requires a voxel grid or analytical 
        # correction, but for simple tracking we approximate or use exact bounds.
        approx_v = (4/3) * np.pi * (r**3)
        current_volume += approx_v 
        
    return spheres




def build_network(spheres, thickness, tunneling_cutoff=1.0):
    G = nx.Graph()
    centers = [s[:3] for s in spheres]
    radii = [s[3] for s in spheres]
    
    # Use a KD-Tree to avoid O(N^2) brute force distance calculations
    tree = KDTree(centers)
    
    # Add crystal nodes
    for i in range(len(spheres)):
        G.add_node(i, z=centers[i][2], r=radii[i])
        
    # Query pairs within a reasonable maximum search radius to find overlaps
    max_search = (max(radii) * 2) + tunneling_cutoff
    pairs = tree.query_pairs(r=max_search)
    
    for i, j in pairs:
        d = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
        # Overlap + quantum tunneling condition
        if d <= (radii[i] + radii[j] + tunneling_cutoff):
            # Calculate physical distance gap for tunneling physics
            gap = max(0, d - (radii[i] + radii[j]))
            G.add_edge(i, j, gap=gap)
            
    # Connect boundaries (Virtual Nodes for electrodes)
    # Node 'Source' at Z=0, Node 'Sink' at Z=Thickness
    G.add_node('Source')
    G.add_node('Sink')
    for i, (x, y, z, r) in enumerate(spheres):
        if z - r <= 0:
            G.add_edge('Source', i, gap=0)
        if z + r >= thickness:
            G.add_edge('Sink', i, gap=0)
            
    # Check if a continuous path actually exists from bottom to top
    has_percolation = nx.has_path(G, 'Source', 'Sink')
    return G, has_percolation




def calculate_resistance(G, rho_bulk, rho_tunnel_basis):
    # 1. Assign Resistances to Edges
    for u, v in G.edges():
        if u in ['Source', 'Sink'] or v in ['Source', 'Sink']:
            G[u][v]['conductance'] = 1e6 # Near-perfect electrode contact
        else:
            gap = G[u][v]['gap']
            # Intra-grain resistance (metallic CuS) + Inter-grain Tunneling (exponential with gap)
            R_intra = rho_bulk / (G.nodes[u]['r'] + G.nodes[v]['r'])
            R_tunnel = rho_tunnel_basis * np.exp(gap / 0.5) # 0.5nm characteristic scale
            
            R_total = R_intra + R_tunnel
            G[u][v]['conductance'] = 1.0 / R_total
            
    # 2. Setup Sparse Conductance Matrix (Kirchhoff's Matrix/Laplacian)
    nodes = [n for n in G.nodes() if n not in ['Source', 'Sink']]
    node_map = {node: idx for idx, node in enumerate(nodes)}
    N = len(nodes)
    
    A = lil_matrix((N, N))
    B = np.zeros(N)
    
    # 3. Apply Node Voltage Equations
    for node in nodes:
        idx = node_map[node]
        total_cond = 0
        for neighbor in G.neighbors(node):
            cond = G[node][neighbor]['conductance']
            total_cond += cond
            if neighbor == 'Source':
                B[idx] += cond * 1.0  # Injecting 1V at Source
            elif neighbor == 'Sink':
                B[idx] += cond * 0.0  # Ground at Sink
            else:
                neighbor_idx = node_map[neighbor]
                A[idx, neighbor_idx] = -cond
        A[idx, idx] = total_cond
        
    # 4. Solve for node voltages V
    V = spsolve(A.tocsr(), B)
    
    # 5. Compute Total Current flowing into the Ground (Sink)
    total_current = 0
    for neighbor in G.neighbors('Sink'):
        neighbor_idx = node_map[neighbor]
        v_neighbor = V[neighbor_idx]
        total_current += (v_neighbor - 0.0) * G['Sink'][neighbor]['conductance']
        
    # R = V / I (where V = 1.0 Volt)
    equivalent_resistance = 1.0 / total_current
    return equivalent_resistance
