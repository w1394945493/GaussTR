import torch


def get_meshgrid(ranges, grid, reso):
    xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
    yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
    zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

    xxx = xxx[:, None, None].expand(*grid)
    yyy = yyy[None, :, None].expand(*grid)
    zzz = zzz[None, None, :].expand(*grid)

    xyz = torch.stack([
        xxx, yyy, zzz
    ], dim=-1)
    return xyz
xyz = get_meshgrid([-50, -50, -5.0, 50, 50, 3.0], [200, 200, 16], 0.5)
print(xyz.shape)