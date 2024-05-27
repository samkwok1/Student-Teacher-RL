import random
import numpy as np
from collections import deque

class Maze():
    def __init__(self, 
                 size: int, 
                 random_seed: int, 
                 verbose: bool) -> None:
        self.size = size
        self.random_seed = random_seed
        self.num_paths = 0
        self.path_lengths = []
        self.verbose = verbose
        # must be after everything else is set
        self.Grid = self.generate_grid()


    def generate_grid(self):

        # Define three helper functions
        # Helper 1: This gets the neighbors of a cell in the grid that are in bounds
        def _get_neighbors(cur_cell):
            x, y = cur_cell
            neighbors = []
            if x > 0:
                neighbors.append((x - 1, y))
            if x < self.size - 1:
                neighbors.append((x + 1, y))
            if y > 0:
                neighbors.append((x, y - 1))
            if y < self.size - 1:
                neighbors.append((x, y + 1))
            return neighbors

        # Helper 2: This counts the number of paths and the lengths of those paths in a grid
        def _count_paths_and_lengths(grid):
            size = len(grid)
            start = (0, 0)
            end = (size - 1, size - 1)
            
            if grid[start[0]][start[1]] == 0 or grid[end[0]][end[1]] == 0:
                return 0, []

            queue = deque([(start, 0, set([start]))])  # Each element is a tuple (current cell, path length, visited cells)
            path_count = 0
            path_lengths = []
            
            while queue:
                cell, length, visited = queue.popleft()
                if cell == end:
                    path_count += 1
                    path_lengths.append(length + 1)
                    continue
                for neighbor in _get_neighbors(cell):
                    if neighbor not in visited and grid[neighbor[0]][neighbor[1]] == 1:
                        new_visited = visited.copy()
                        new_visited.add(neighbor)
                        queue.append((neighbor, length + 1, new_visited))
            
            return path_count, path_lengths
        
        # Helper 3: This is part of Prim's algorithm for maze generation, gets the number of unexplored neighbors
        def _get_unexplored_neighbors(cur_cell, Grid):
            neighbors = _get_neighbors(cur_cell)
            unexplored = []
            for neighbor in neighbors:
                loc_x, loc_y = neighbor
                if Grid[loc_x][loc_y] == 0:
                    unexplored.append((loc_x, loc_y))

            # If the cell has 2 or more explore neighbors, do nothing
            if len(neighbors) - len(unexplored) >= 2:
                return []
            return unexplored
        
        # Run the functionality
        # set the random seed
        random.seed(self.random_seed)

        # Initialize the grid
        Grid = np.zeros((self.size, self.size))

        # We always want the exit to be at the bottom right
        cur_cell = (self.size - 1, self.size - 1)
        start_cell = (0, 0)


        all_cells = [cur_cell, start_cell]
        while all_cells:
            neighbors = _get_unexplored_neighbors(cur_cell, Grid)
            if neighbors:
                Grid[cur_cell[0]][cur_cell[1]] = 1
                for neighbor in neighbors:
                    all_cells.append(neighbor)
            all_cells.remove(cur_cell)
            if all_cells:
                cur_cell = all_cells[random.randint(0, len(all_cells) - 1)]

        modifications = ['3,6,1','4,2,0', '4,1,1', '6,0,1']
        for modification in modifications:
            x, y, val = modification.split(',')
            Grid[int(x)][int(y)] = int(val)

        num_paths, path_lengths = _count_paths_and_lengths(Grid.copy())

        self.num_paths = num_paths
        self.path_lengths = path_lengths

        if self.verbose:
            print("Maze generated:")
            print(Grid)

        return Grid

# def main():
#     maze = Maze(9, 5)

# if __name__ == "__main__":
#     main()

            
            



