from algorithm.puzzle import Puzzle

initial = tuple(map(int, input("Input: ").strip()))
goal = (1, 3, 5, 7, 0, 2, 6, 8, 4)
puzzle = Puzzle(initial, goal)
print(puzzle.astar())