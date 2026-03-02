import numpy as np
from scipy.spatial import KDTree
from sklearn.neighbors import kneighbors_graph
import networkx as nx
from magnum import Range3D
class PRMPlanner:
    def __init__(self, bounds, num_samples, obstacle_func, k=10, scene_id=None):
        """
        Initialize a Probabilistic Roadmap (PRM) planner.

        Args:
            bounds: Workspace bounds, shaped like ((min_x, max_x), (min_y, max_y), (min_z, max_z)).
            num_samples: Number of random samples to draw.
            obstacle_func: Obstacle checking function. Takes a point and scene_id, returns True if in obstacle.
            k: Number of nearest neighbors used when building the roadmap.
        """
        self._bounds = None
        self.bounds = bounds
        self.num_samples = num_samples
        self.obstacle_func = obstacle_func
        self.k = k
        self.samples = []
        self.graph = nx.Graph()

        self.scene_id = scene_id


    def sample_free_space(self):
        """Sample random points in free space."""
        while len(self.samples) < self.num_samples:
            point = np.random.uniform([b[0] for b in self._bounds], [b[1] for b in self._bounds])
            if not self.obstacle_func(point, self.scene_id):
                self.samples.append(point)
        self.samples = np.array(self.samples)

    def pre_plan(self):
        self.sample_free_space()
        self.build_roadmap()

    def build_roadmap(self):
        """Build the roadmap graph."""
        # Use KDTree to speed up neighbor search
        tree = KDTree(self.samples)
        # Build a k-nearest-neighbor graph
        A = kneighbors_graph(self.samples, self.k, mode='distance', metric='euclidean', include_self=False)
        A = A.toarray()
        
        # Add edges into the graph
        for i, neighbors in enumerate(A):
            for j, dist in enumerate(neighbors):
                # Ensure the midpoint of the edge is not inside an obstacle
                if dist > 0 and not self.obstacle_func((self.samples[i] + self.samples[j]) / 2, self.scene_id):
                    self.graph.add_edge(i, j, weight=dist)

    def plan(self, start, goal):
        """Plan a path from start to goal."""
        # Add start and goal into the graph
        start_idx = len(self.samples)
        goal_idx = start_idx + 1
        self.samples = np.vstack([self.samples, start, goal])
        self.graph.add_node(start_idx)
        self.graph.add_node(goal_idx)
        
        # Connect start and goal to existing graph
        tree = KDTree(self.samples)
        start_neighbors = tree.query(start, self.k)[1]
        goal_neighbors = tree.query(goal, self.k)[1]
        
        for neighbor in start_neighbors:
            if not self.obstacle_func((start + self.samples[neighbor]) / 2, self.scene_id):
                self.graph.add_edge(start_idx, neighbor, weight=np.linalg.norm(start - self.samples[neighbor]))
        for neighbor in goal_neighbors:
            if not self.obstacle_func((goal + self.samples[neighbor]) / 2, self.scene_id):
                self.graph.add_edge(goal_idx, neighbor, weight=np.linalg.norm(goal - self.samples[neighbor]))
        
        # Use A* to search for the shortest path
        path = nx.astar_path(self.graph, start_idx, goal_idx, heuristic=lambda a, b: np.linalg.norm(self.samples[a] - self.samples[b]))
        return [self.samples[i] for i in path]

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        if isinstance(bounds, Range3D):
            self._bounds = np.array([bounds.min, bounds.max]).T
        else:
            self._bounds = bounds

    def debug():
    def draw_ball(ax, data):
        center = data[:3]
        radius = data[3]
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        color = cm.coolwarm(z/7)
        ax.plot_surface(x, y, z, facecolors=color, alpha=0.5)

        
    bounds = np.array([[0,0,0],[10,5,5]])
    ball_num = 20
    
    def ball_generate(num):
        pos = np.random.uniform(bounds[0], bounds[1], [num, 3])
        r = np.random.uniform(0.5,1,num)
        return np.c_[pos, r]
    balls = ball_generate(ball_num)
    
    # Example: define an obstacle checking function
    def is_obstacle(point, scene_id=None):
        is_ob = False
        # Assume a spherical obstacle at the center of the space
        for data in balls:
            center = data[:3]
            radius = data[3]
            if np.linalg.norm(point - center) < radius:
                is_ob = True
                break
            
        return is_ob 

    import time
    
    # Create PRM planner instance
    start = time.time()
    planner = PRMPlanner(bounds=((0, 10), (0, 10), (0, 10)), num_samples=500, obstacle_func=is_obstacle, k=10,scene_id=0)
    planner.pre_plan()
    # planner.sample_free_space()
    # planner.build_roadmap()
    end = time.time()
    print(f"time:{end-start}%5f")
    # Plan path
    start = np.array([1, 1, 1])
    goal = np.array([9, 3, 3])
    path = planner.plan(start, goal)

    print("Path from start to goal:", path)

    from matplotlib import pyplot as plt
    from matplotlib import cm
    # plot 3d path
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for data in balls:
        draw_ball(ax, data)
    # plot 3d path
    path = np.array(path)  # convert path to numpy array for easier slicing

    ax.plot(path[:,0], path[:,1], path[:,2], linewidth=3, color="red")
    ax.set_aspect("equal")
    plt.show()
    fig.savefig('debug/pathfinder_demo.png')
    
if __name__ == "__main__":
    debug()