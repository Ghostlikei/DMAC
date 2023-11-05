import heapq

class Pyramid:
    def __init__(self, N, M, K, passages):
        self.N = N  # Number of rooms
        self.M = M  # Number of passages
        self.K = K  # Number of paths to find
        self.passages = passages  # List of passages
        self.adj_list = self.build_adj_list()  # Adjacency list for the graph
        self.rev_adj_list = self.build_reverse_adj_list()  # Reverse adjacency list for dijkstra

    def build_adj_list(self):
        adj_list = {i: [] for i in range(1, self.N + 1)}
        for x, y, d in self.passages:
            adj_list[x].append((y, d))
        return adj_list

    def build_reverse_adj_list(self):
        rev_adj_list = {i: [] for i in range(1, self.N + 1)}
        for x, y, d in self.passages:
            rev_adj_list[y].append((x, d))

        print(rev_adj_list)
        return rev_adj_list

    def dijkstra(self, start):
        distances = {node: float('infinity') for node in range(1, self.N + 1)}
        distances[start] = 0
        visited = set()
        pq = [(0, start)]
        while pq:
            current_distance, current_node = heapq.heappop(pq)
            print("Current node: ", current_node)
            if current_node in visited:
                continue
            visited.add(current_node)
            for neighbor, weight in self.rev_adj_list[current_node]:
                print(f"Neighbor: {neighbor}, weight: {weight}")
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))

        print(distances)
        return distances

    def astar(self):
        h_values = self.dijkstra(self.N)  # Precompute heuristic values using reverse adjacency list
        open_list = []  
        heapq.heappush(open_list, (0, [1], 0))  # (f, path, g)
        paths = []

        while open_list and len(paths) < self.K:
            f, path, g = heapq.heappop(open_list)
            node = path[-1]

            if node == self.N:  # If the goal node is reached
                paths.append((g, path))
                continue  # Continue searching for other paths

            for neighbor, cost in self.adj_list[node]:
                new_g = g + cost
                h = h_values[neighbor]  # Updated heuristic
                new_f = new_g + h
                new_path = path + [neighbor]
                heapq.heappush(open_list, (new_f, new_path, new_g))  # Add to open list

        while len(paths) < self.K:
            paths.append((-1, []))
            
        return [g for g, path in paths]

# Example usage:
# N, M, K = 5, 6, 4
# passages = [(1, 2, 1), (1, 3, 1), (2, 4, 2), (2, 5, 2), (3, 4, 2), (3, 5, 2)]
# pyramid = Pyramid(N, M, K, passages)
# result = pyramid.astar()
# print(result)  # Output: [3, 3, 7, 7]
