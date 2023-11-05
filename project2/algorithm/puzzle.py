from queue import PriorityQueue

class Puzzle:
    def __init__(self, initial, goal):
        self.initial = initial
        self.goal = goal

    def astar(self):
        open_list = PriorityQueue()
        open_list.put((self.h(self.initial), self.initial))
        g = {self.initial: 0}
        parent = {self.initial: None}

        while not open_list.empty():
            _, current = open_list.get()
            if current == self.goal:
                return g[current]

            for neighbor in self.get_neighbors(current):
                tentative_g = g[current] + 1
                if neighbor not in g or tentative_g < g[neighbor]:
                    g[neighbor] = tentative_g
                    f = tentative_g + self.h(neighbor)
                    open_list.put((f, neighbor))
                    parent[neighbor] = current

        return None

    def h(self, state):
        return sum(abs(b % 3 - g % 3) + abs(b // 3 - g // 3) for b, g in ((state.index(i), self.goal.index(i)) for i in range(1, 9)))

    def get_neighbors(self, state):
        neighbors = []
        i = state.index(0)
        rows = [0, -1, 0, 1]
        cols = [-1, 0, 1, 0]

        for j in range(4):
            x, y = i % 3 + rows[j], i // 3 + cols[j]
            if 0 <= x < 3 and 0 <= y < 3:
                neighbor = list(state)
                neighbor[i], neighbor[x + y * 3] = neighbor[x + y * 3], neighbor[i]
                neighbors.append(tuple(neighbor))

        return neighbors
