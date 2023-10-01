import numpy as np
import torch


def create_grid(nc: int, offset=0.5) -> torch.Tensor:
    """
    Create xy values of nc x nc grid
    offset (float): offset in units of original 256 x 256 image
                    offset 0 and nc 256 give identity mapping
                    Use offset 0.5 for shifted contrail label
    Returns: grid (Tensor)
      function generates a 2D grid of (x, y) coordinates in the range [-1, 1] for torch grid_sample()
    """
    # third dimension [2] represents the (x, y) coordinates of each grid point.
    grid = np.zeros((nc, nc, 2), dtype=np.float32)
    # grid.shape => (512, 512, 2)
    
    # loops are used to fill in the grid with (x, y) coordinates.
    for ix in range(nc):
        for iy in range(nc):
            # calculates the y-coordinate of the grid point at position (ix, iy).            
            # 2 * (ix + 0.5) / nc => scales the y-coordinate based on the current row index (ix) within the grid.
            # a positive offset value will move the grid points upward.
            grid[ix, iy, 1] = -1 + (2 * (ix + 0.5) / nc) + (offset / 128)
            # ix, iy, grid[ix, iy, 1] => 1, 2, -0.9902344

            # calculates the x-coordinate of the grid point at position (ix, iy).
            grid[ix, iy, 0] = -1 + (2 * (iy + 0.5) / nc) + (offset / 128)

            # ix, iy, grid[ix, iy, 1] => 1, 2, -0.9863281
    
    # grid[0][:5] => [[-0.9941406 -0.9941406] [-0.9902344 -0.9941406] [-0.9863281 -0.9941406] [-0.9824219 -0.9941406] [-0.9785156 -0.9941406]]

    grid = torch.from_numpy(grid).unsqueeze(0)
    # grid.shape => torch.Size([1, 512, 512, 2])
    return grid
