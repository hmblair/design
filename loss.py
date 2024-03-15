
import torch

def kabsch_align(x : torch.Tensor, y : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Align the point clouds in x with the point clouds in y using the 
    Kabsch algorithm, which finds the rotation matrix that aligns the two point
    clouds, such that the root mean square deviation between the two point
    clouds is minimized.

    Parameters:
    -----------
    x : torch.Tensor
        A tensor of shape (*, N, 3) representing the first batch of point clouds, 
        where * is any number of dimensions, generally representing the batch 
        dimension.
    y : torch.Tensor
        A tensor of shape (*, N, 3) representing the second batch of point clouds.

    Returns:
    --------
    torch.Tensor
        A tensor of shape (*, N, 3) representing the first point cloud optimally
        aligned with the second point cloud.
    torch.Tensor
        A tensor of shape (*, N, 3) representing the second point cloud, with no
        change in its position aside from a zero-centering.
    """

    # center the point clouds
    x = x - x.mean(1, keepdim=True)
    y = y - y.mean(1, keepdim=True)

    # calculate the covariance matrix
    C = torch.einsum('...ij,...ik->...jk', x, y)

    # calculate the singular value decomposition
    U, _, V = torch.linalg.svd(C)

    # calculate the optimal rotation matrix
    R = torch.einsum('...ij,...jk->...ik', U, V)

    # apply the rotation matrix to the first point cloud, and return it,
    # along with the zero-centered second point cloud
    return torch.einsum('...ij,...jk->...ik', x, R), y



class KabschLoss(torch.nn.Module):
    """
    The Kabsch loss function, which calculates the mean square deviation
    between two point clouds after applying the Kabsch algorithm to optimally
    align the point clouds.
    """
    def __init__(self):
        super().__init__()


    def forward(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        """
        Calculate the mean square deviation between two point clouds, after
        applying the Kabsch algorithm to optimally align the point clouds.

        Parameters:
        -----------
        x : torch.Tensor
            A tensor of shape (N, 3) representing the first point cloud.
        y : torch.Tensor
            A tensor of shape (N, 3) representing the second point cloud.

        Returns:
        --------
        torch.Tensor
            A tensor of shape (1,) representing the mean square deviation
            between the two point clouds.
        """
        # align the point clouds
        x, y = kabsch_align(x, y)

        # calculate the mean square deviation
        return ((x - y) ** 2).mean()