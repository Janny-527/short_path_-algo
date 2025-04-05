import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from heapq import heappush, heappop
from matplotlib.widgets import TextBox, Button
from matplotlib.patches import Rectangle

class RoadNetworkSimulator:
    def __init__(self, num_nodes=20, connectivity=0.2, seed=42):
        """Initialize the road network simulator with a random graph."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Create graph
        self.G = nx.random_geometric_graph(num_nodes, connectivity, seed=seed)
        
        # Assign random weights to edges (distances)
        for u, v in self.G.edges():
            self.G[u][v]['weight'] = np.round(np.random.uniform(1, 10), 2)
            self.G[u][v]['original_weight'] = self.G[u][v]['weight']
        
        # Ensure graph is connected
        if not nx.is_connected(self.G):
            components = list(nx.connected_components(self.G))
            for i in range(1, len(components)):
                u = random.choice(list(components[0]))
                v = random.choice(list(components[i]))
                self.G.add_edge(u, v, weight=np.round(np.random.uniform(1, 10), 2))
                self.G[u][v]['original_weight'] = self.G[u][v]['weight']
        
        # Assign positions for nodes (for visualization)
        self.pos = nx.get_node_attributes(self.G, 'pos')
        
        # For storing results
        self.algorithm_results = {}
    
    def apply_traffic_conditions(self, traffic_factor=0.5):
        """Apply random traffic conditions to change edge weights."""
        for u, v in self.G.edges():
            # Increase weight by a random factor (simulating traffic)
            traffic_multiplier = 1 + random.uniform(0, traffic_factor)
            self.G[u][v]['weight'] = self.G[u][v]['original_weight'] * traffic_multiplier
    
    def reset_traffic_conditions(self):
        """Reset all edge weights to their original values."""
        for u, v in self.G.edges():
            self.G[u][v]['weight'] = self.G[u][v]['original_weight']
    
    def dijkstra(self, source, target):
        """Implementation of Dijkstra's algorithm."""
        start_time = time.time()
        
        distances = {node: float('infinity') for node in self.G.nodes()}
        previous = {node: None for node in self.G.nodes()}
        distances[source] = 0
        pq = [(0, source)]
        
        while pq:
            current_distance, current_node = heappop(pq)
            
            # If we reached the target node
            if current_node == target:
                break
                
            # If we've already found a better path
            if current_distance > distances[current_node]:
                continue
                
            # Check all neighbors
            for neighbor in self.G.neighbors(current_node):
                weight = self.G[current_node][neighbor]['weight']
                distance = current_distance + weight
                
                # If we found a better path
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current_node
                    heappush(pq, (distance, neighbor))
        
        # Reconstruct path
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        
        end_time = time.time()
        
        return {
            'algorithm': 'Dijkstra',
            'path': path if path[0] == source else [],  # Empty if no path found
            'distance': distances[target] if target in distances else float('infinity'),
            'time': end_time - start_time
        }
    
    def a_star(self, source, target):
        """Implementation of A* algorithm using straight-line distance heuristic."""
        start_time = time.time()
        
        # Heuristic function: Euclidean distance
        def heuristic(n1, n2):
            x1, y1 = self.pos[n1]
            x2, y2 = self.pos[n2]
            return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        # A* algorithm
        open_set = [(0, source)]  # Priority queue with f-score and node
        came_from = {node: None for node in self.G.nodes()}
        g_score = {node: float('infinity') for node in self.G.nodes()}
        g_score[source] = 0
        f_score = {node: float('infinity') for node in self.G.nodes()}
        f_score[source] = heuristic(source, target)
        
        while open_set:
            _, current = heappop(open_set)
            
            if current == target:
                break
                
            for neighbor in self.G.neighbors(current):
                tentative_g_score = g_score[current] + self.G[current][neighbor]['weight']
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, target)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        # Reconstruct path
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        
        end_time = time.time()
        
        return {
            'algorithm': 'A*',
            'path': path if path[0] == source else [],  # Empty if no path found
            'distance': g_score[target] if target in g_score else float('infinity'),
            'time': end_time - start_time
        }
    
    def bellman_ford(self, source, target):
        """Implementation of Bellman-Ford algorithm."""
        start_time = time.time()
        
        # Initialize distances and predecessors
        distances = {node: float('infinity') for node in self.G.nodes()}
        predecessors = {node: None for node in self.G.nodes()}
        distances[source] = 0
        
        # Relax edges repeatedly
        for _ in range(len(self.G.nodes()) - 1):
            for u, v in self.G.edges():
                if distances[u] + self.G[u][v]['weight'] < distances[v]:
                    distances[v] = distances[u] + self.G[u][v]['weight']
                    predecessors[v] = u
                if distances[v] + self.G[u][v]['weight'] < distances[u]:  # Since our graph is undirected
                    distances[u] = distances[v] + self.G[u][v]['weight']
                    predecessors[u] = v
        
        # Check for negative weight cycles (which shouldn't exist in a road network)
        for u, v in self.G.edges():
            if distances[u] + self.G[u][v]['weight'] < distances[v]:
                print("Warning: Graph contains a negative weight cycle")
                break
        
        # Reconstruct path
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()
        
        end_time = time.time()
        
        return {
            'algorithm': 'Bellman-Ford',
            'path': path if path[0] == source else [],  # Empty if no path found
            'distance': distances[target] if target in distances else float('infinity'),
            'time': end_time - start_time
        }
    
    def find_shortest_paths(self, source, target):
        """Find shortest paths using all algorithms and store results."""
        # Reset results
        self.algorithm_results = {}
        
        # Run algorithms
        self.algorithm_results['Dijkstra'] = self.dijkstra(source, target)
        self.algorithm_results['A*'] = self.a_star(source, target)
        self.algorithm_results['Bellman-Ford'] = self.bellman_ford(source, target)
        
        return self.algorithm_results
    
    def visualize_graph(self):
        """Visualize the graph with weighted edges."""
        plt.figure(figsize=(12, 8))
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(self.G, self.pos, node_size=300, node_color='skyblue')
        nx.draw_networkx_labels(self.G, self.pos)
        
        # Draw edges with weights
        edge_labels = {(u, v): f"{data['weight']:.2f}" for u, v, data in self.G.edges(data=True)}
        nx.draw_networkx_edges(self.G, self.pos, width=1.0, alpha=0.7)
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, font_size=8)
        
        plt.title("Road Network Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def visualize_paths(self, source, target):
        """Visualize the shortest paths found by all algorithms."""
        if not self.algorithm_results:
            print("No paths to visualize. Run 'find_shortest_paths' first.")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, alg_name in enumerate(['Dijkstra', 'A*', 'Bellman-Ford']):
            ax = axes[i]
            result = self.algorithm_results[alg_name]
            path = result['path']
            
            # Draw the base graph
            nx.draw_networkx_nodes(self.G, self.pos, node_size=200, node_color='lightgray', ax=ax)
            nx.draw_networkx_edges(self.G, self.pos, width=1.0, alpha=0.3, ax=ax)
            
            # Highlight the path
            if path:
                path_edges = list(zip(path, path[1:]))
                nx.draw_networkx_nodes(self.G, self.pos, nodelist=[source, target], 
                                    node_size=300, node_color='green', ax=ax)
                nx.draw_networkx_nodes(self.G, self.pos, nodelist=path[1:-1], 
                                    node_size=200, node_color='orange', ax=ax)
                nx.draw_networkx_edges(self.G, self.pos, edgelist=path_edges, 
                                    width=2.0, edge_color='red', ax=ax)
            
            # Add labels
            nx.draw_networkx_labels(self.G, self.pos, font_size=8, ax=ax)
            
            ax.set_title(f"{alg_name}\nTime: {result['time']:.6f}s\nDistance: {result['distance']:.2f}")
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_visualization(self):
        """Create an interactive visualization for path finding."""
        self.fig, (self.ax, self.results_ax) = plt.subplots(1, 2, figsize=(16, 8), 
                                                         gridspec_kw={'width_ratios': [2, 1]})
        
        # Draw the graph
        nx.draw_networkx_nodes(self.G, self.pos, node_size=300, node_color='skyblue', ax=self.ax)
        nx.draw_networkx_edges(self.G, self.pos, width=1.0, alpha=0.7, ax=self.ax)
        nx.draw_networkx_labels(self.G, self.pos, ax=self.ax)
        
        # Add edge weights
        edge_labels = {(u, v): f"{data['weight']:.1f}" for u, v, data in self.G.edges(data=True)}
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, font_size=7, ax=self.ax)
        
        # Set up text boxes for source and target input
        plt.subplots_adjust(bottom=0.2)
        source_ax = plt.axes([0.15, 0.05, 0.1, 0.075])
        target_ax = plt.axes([0.35, 0.05, 0.1, 0.075])
        self.source_box = TextBox(source_ax, 'Source Node:', initial='0')
        self.target_box = TextBox(target_ax, 'Target Node:', initial='10')
        
        # Add buttons
        find_path_ax = plt.axes([0.5, 0.05, 0.1, 0.075])
        apply_traffic_ax = plt.axes([0.65, 0.05, 0.15, 0.075])
        reset_traffic_ax = plt.axes([0.85, 0.05, 0.1, 0.075])
        
        self.find_path_button = Button(find_path_ax, 'Find Paths')
        self.find_path_button.on_clicked(self.on_find_paths_click)
        
        self.apply_traffic_button = Button(apply_traffic_ax, 'Apply Traffic')
        self.apply_traffic_button.on_clicked(self.on_apply_traffic_click)
        
        self.reset_traffic_button = Button(reset_traffic_ax, 'Reset Traffic')
        self.reset_traffic_button.on_clicked(self.on_reset_traffic_click)
        
        # Clear results area
        self.results_ax.axis('off')
        self.results_ax.text(0.5, 0.95, "Algorithm Results", 
                           horizontalalignment='center', fontsize=14, fontweight='bold')
        
        self.ax.set_title("Road Network Simulator")
        self.ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def on_find_paths_click(self, event):
        """Handle the Find Paths button click."""
        try:
            source = int(self.source_box.text)
            target = int(self.target_box.text)
            
            if source not in self.G.nodes() or target not in self.G.nodes():
                self.display_error("Invalid node IDs. Please use existing node numbers.")
                return
            
            # Find shortest paths using all algorithms
            results = self.find_shortest_paths(source, target)
            
            # Update visualization
            self.ax.clear()
            
            # Draw the base graph
            nx.draw_networkx_nodes(self.G, self.pos, node_size=300, node_color='lightgray', ax=self.ax)
            nx.draw_networkx_edges(self.G, self.pos, width=1.0, alpha=0.3, ax=self.ax)
            nx.draw_networkx_labels(self.G, self.pos, ax=self.ax)
            
            # Highlight source and target
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=[source, target], 
                                 node_size=400, node_color='green', ax=self.ax)
            
            # Draw paths with different colors
            colors = ['red', 'blue', 'purple']
            offsets = [0.02, 0, -0.02]  # For visual separation of paths
            
            for i, alg_name in enumerate(['Dijkstra', 'A*', 'Bellman-Ford']):
                result = results[alg_name]
                path = result['path']
                
                if path:
                    # Draw path with offset for visibility
                    path_edges = list(zip(path, path[1:]))
                    edge_pos = {}
                    for u, v in path_edges:
                        pu = self.pos[u]
                        pv = self.pos[v]
                        # Create slight offset perpendicular to edge
                        dx = pv[0] - pu[0]
                        dy = pv[1] - pu[1]
                        length = np.sqrt(dx*dx + dy*dy)
                        offset = offsets[i]
                        edge_pos[(u, v)] = [
                            (pu[0] + offset*dy/length, pu[1] - offset*dx/length),
                            (pv[0] + offset*dy/length, pv[1] - offset*dx/length)
                        ]
                    
                    # Draw path with specified color
                    for edge, pos in edge_pos.items():
                        self.ax.plot([pos[0][0], pos[1][0]], [pos[0][1], pos[1][1]], 
                                   color=colors[i], linewidth=2.5, alpha=0.8)
            
            # Update results panel
            self.display_results(results, source, target)
            
            self.ax.set_title(f"Road Network - Paths from {source} to {target}")
            self.ax.axis('off')
            self.fig.canvas.draw_idle()
            
        except ValueError:
            self.display_error("Please enter valid node numbers.")
    
    def on_apply_traffic_click(self, event):
        """Handle the Apply Traffic button click."""
        self.apply_traffic_conditions(traffic_factor=0.8)
        
        # Redraw graph with new weights
        self.ax.clear()
        nx.draw_networkx_nodes(self.G, self.pos, node_size=300, node_color='skyblue', ax=self.ax)
        nx.draw_networkx_edges(self.G, self.pos, width=1.0, alpha=0.7, ax=self.ax)
        nx.draw_networkx_labels(self.G, self.pos, ax=self.ax)
        
        # Update edge labels
        edge_labels = {(u, v): f"{data['weight']:.1f}" for u, v, data in self.G.edges(data=True)}
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, font_size=7, ax=self.ax)
        
        self.ax.set_title("Road Network - Traffic Applied")
        self.ax.axis('off')
        self.fig.canvas.draw_idle()
        
        # Display message in results panel
        self.results_ax.clear()
        self.results_ax.text(0.5, 0.95, "Traffic Applied", 
                           horizontalalignment='center', fontsize=14, fontweight='bold')
        self.results_ax.text(0.5, 0.85, "Road weights have increased due to traffic.\nTry finding paths again.", 
                           horizontalalignment='center', fontsize=12)
        self.results_ax.axis('off')
        self.fig.canvas.draw_idle()
    
    def on_reset_traffic_click(self, event):
        """Handle the Reset Traffic button click."""
        self.reset_traffic_conditions()
        
        # Redraw graph with original weights
        self.ax.clear()
        nx.draw_networkx_nodes(self.G, self.pos, node_size=300, node_color='skyblue', ax=self.ax)
        nx.draw_networkx_edges(self.G, self.pos, width=1.0, alpha=0.7, ax=self.ax)
        nx.draw_networkx_labels(self.G, self.pos, ax=self.ax)
        
        # Update edge labels
        edge_labels = {(u, v): f"{data['weight']:.1f}" for u, v, data in self.G.edges(data=True)}
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, font_size=7, ax=self.ax)
        
        self.ax.set_title("Road Network - Traffic Reset")
        self.ax.axis('off')
        self.fig.canvas.draw_idle()
        
        # Display message in results panel
        self.results_ax.clear()
        self.results_ax.text(0.5, 0.95, "Traffic Reset", 
                           horizontalalignment='center', fontsize=14, fontweight='bold')
        self.results_ax.text(0.5, 0.85, "Road weights have been reset to original values.", 
                           horizontalalignment='center', fontsize=12)
        self.results_ax.axis('off')
        self.fig.canvas.draw_idle()
    
    def display_results(self, results, source, target):
        """Display algorithm results in the results panel."""
        self.results_ax.clear()
        
        # Title
        self.results_ax.text(0.5, 0.95, f"Paths from {source} to {target}", 
                           horizontalalignment='center', fontsize=14, fontweight='bold')
        
        # Create a color legend
        self.results_ax.add_patch(Rectangle((0.05, 0.85), 0.03, 0.03, color='red'))
        self.results_ax.text(0.1, 0.85, "Dijkstra", fontsize=10)
        
        self.results_ax.add_patch(Rectangle((0.35, 0.85), 0.03, 0.03, color='blue'))
        self.results_ax.text(0.4, 0.85, "A*", fontsize=10)
        
        self.results_ax.add_patch(Rectangle((0.65, 0.85), 0.03, 0.03, color='purple'))
        self.results_ax.text(0.7, 0.85, "Bellman-Ford", fontsize=10)
        
        # Display results for each algorithm
        y_pos = 0.75
        for alg_name in ['Dijkstra', 'A*', 'Bellman-Ford']:
            result = results[alg_name]
            path_str = ' → '.join(str(node) for node in result['path']) if result['path'] else "No path found"
            
            self.results_ax.text(0.5, y_pos, alg_name, 
                               horizontalalignment='center', fontsize=12, fontweight='bold')
            y_pos -= 0.05
            
            self.results_ax.text(0.5, y_pos, f"Time: {result['time']:.6f} seconds", 
                               horizontalalignment='center', fontsize=10)
            y_pos -= 0.05
            
            self.results_ax.text(0.5, y_pos, f"Distance: {result['distance']:.2f}", 
                               horizontalalignment='center', fontsize=10)
            y_pos -= 0.05
            
            # Truncate long paths for display
            if len(path_str) > 60:
                path_str = path_str[:57] + "..."
            
            self.results_ax.text(0.5, y_pos, f"Path: {path_str}", 
                               horizontalalignment='center', fontsize=9)
            y_pos -= 0.07
        
        self.results_ax.axis('off')
    
    def display_error(self, message):
        """Display an error message in the results panel."""
        self.results_ax.clear()
        self.results_ax.text(0.5, 0.5, "Error", 
                           horizontalalignment='center', fontsize=14, fontweight='bold', color='red')
        self.results_ax.text(0.5, 0.4, message, 
                           horizontalalignment='center', fontsize=12)
        self.results_ax.axis('off')
        self.fig.canvas.draw_idle()

    def run_algorithm_benchmark(self, num_trials=10):
        """Run a benchmark comparing all algorithms on various source-target pairs."""
        results = {
            'Dijkstra': {'times': [], 'distances': []},
            'A*': {'times': [], 'distances': []},
            'Bellman-Ford': {'times': [], 'distances': []}
        }
        
        nodes = list(self.G.nodes())
        
        print("Running benchmark...")
        for _ in range(num_trials):
            source = random.choice(nodes)
            target = random.choice([n for n in nodes if n != source])
            
            paths = self.find_shortest_paths(source, target)
            
            for alg in ['Dijkstra', 'A*', 'Bellman-Ford']:
                results[alg]['times'].append(paths[alg]['time'])
                results[alg]['distances'].append(paths[alg]['distance'])
        
        # Print results
        print("\nBenchmark Results (Average of {} trials):".format(num_trials))
        print("{:<15} {:<15} {:<15}".format("Algorithm", "Avg Time (s)", "Avg Distance"))
        print("-" * 45)
        
        for alg in ['Dijkstra', 'A*', 'Bellman-Ford']:
            avg_time = np.mean(results[alg]['times'])
            avg_dist = np.mean(results[alg]['distances'])
            print("{:<15} {:<15.6f} {:<15.2f}".format(alg, avg_time, avg_dist))
        
        # Plot time comparison
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        algs = ['Dijkstra', 'A*', 'Bellman-Ford']
        avg_times = [np.mean(results[alg]['times']) for alg in algs]
        plt.bar(algs, avg_times, color=['blue', 'green', 'red'])
        plt.title('Average Execution Time')
        plt.ylabel('Time (seconds)')
        plt.yscale('log')  # Log scale to better see differences
        
        plt.subplot(1, 2, 2)
        for alg in algs:
            plt.plot(results[alg]['times'], label=alg)
        plt.title('Execution Time per Trial')
        plt.xlabel('Trial')
        plt.ylabel('Time (seconds)')
        plt.legend()
        plt.yscale('log')  # Log scale to better see differences
        
        plt.tight_layout()
        plt.show()
        
        return results

# Demo script to run the road network simulator
def run_demo():
    # Create simulator with 20 nodes
    simulator = RoadNetworkSimulator(num_nodes=20, connectivity=0.2)
    
    print(f"Road network created with {len(simulator.G.nodes())} nodes and {len(simulator.G.edges())} edges")
    
    # Display the graph
    simulator.visualize_graph()
    
    # Run interactive visualization
    simulator.create_interactive_visualization()
    
    # Alternatively, run a benchmark comparison
    # simulator.run_algorithm_benchmark(num_trials=20)

if __name__ == "__main__":
    run_demo()