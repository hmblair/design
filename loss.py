
import torch
import torch.nn as nn

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
    x = x - x.mean(-2, keepdim=True)
    y = y - y.mean(-2, keepdim=True)

    # calculate the covariance matrix
    C = torch.einsum('...ij,...ik->...jk', x, y)

    # calculate the singular value decomposition
    U, _, V = torch.linalg.svd(C)

    # calculate the optimal rotation matrix
    R = torch.einsum('...ij,...jk->...ik', U, V)

    # apply the rotation matrix to the first point cloud, and return it,
    # along with the zero-centered second point cloud
    return torch.einsum('...ij,...jk->...ik', x, R), y



class KabschLoss(nn.Module):
    """
    The Kabsch loss function, which calculates the mean square deviation
    between two point clouds after applying the Kabsch algorithm to optimally
    align the point clouds.
    """
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

        # align the point clouds using the Kabsch algorithm
        x, y = kabsch_align(x, y)

        # calculate the mean square deviation
        return ((x - y) ** 2).mean()
    


def frame_align(x : torch.Tensor, R : torch.Tensor) -> torch.Tensor:
    """
    Take a point cloud x in R^d and corresponding orientations R, and align the point
    cloud at each point using the rotation matrix at that point. The point cloud
    is also centered at the origin.

    Parameters:
    -----------
    x : torch.Tensor
        A tensor of shape (..., N, d) representing the point cloud.
    R : torch.Tensor
        A tensor of shape (..., N, d, d) representing the frame rotation matrix 
        at each point in the point cloud.

    Returns:
    --------
    torch.Tensor
        A tensor of shape (..., N, N, d) representing the point cloud aligned at
        each point using the rotation matrix at that point. Dimension -d 
        corresponds to which frame was used to align the point cloud, and 
        dimension -d corresponds to the point in the point cloud after alignment.
    """

    # center the point cloud
    x = x - x.mean(-2, keepdim=True)

    # invert the rotation matrices by transposing them
    R_inv = R.transpose(-2, -1)

    # align the point cloud at each point separately
    return torch.einsum('...lij,...kj->...lki', R_inv, x)



class FrameAlignedPointError(nn.Module):
    """
    For copmuting the Frame Aligned Point Error (FAPE) between two oriented point
    clouds.
    """
    def forward(
            self, 
            x : torch.Tensor, 
            R_x : torch.Tensor, 
            y : torch.Tensor, 
            R_y : torch.Tensor,
            ) -> torch.Tensor:
        """
        Compute the NxN matrix of frame-aligned distances between the two point
        clouds in R^d, and return the mean such distance. 

        Parameters:
        -----------
        x : torch.Tensor
            A tensor of shape (..., d, 3) representing the first point cloud.
        R_x : torch.Tensor
            A tensor of shape (..., N, d, d) representing the orientation of each
            point in the first point cloud, stored as a rotation matrix.
        y : torch.Tensor
            A tensor of shape (..., N, d) representing the second point cloud.
        R_y : torch.Tensor
            A tensor of shape (..., N, d, d) representing the orientation of each
            point in the second point cloud, stored as a rotation matrix.
        """

        # align the point clouds according to their rotations
        x_aligned = frame_align(x, R_x)
        y_aligned = frame_align(y, R_y)

        # calculate the frame-aligned point error, which is the mean squared
        # distance between the two point clouds
        return ((x_aligned - y_aligned) ** 2).mean()