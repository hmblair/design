
import torch
import dgl

def graph_via_threshold(x : torch.Tensor, t : float) -> dgl.DGLGraph:
    """
    Take a tensor x representing a point cloud, and a threshold t representing
    the maximum distance between points for them to be connected by an edge,
    and return a graph where the nodes are the points in the point cloud and
    the edges are the connections between points that are closer than the
    threshold. Self-connections are ignored. The graph will also contain edge 
    features representing the relative position of the nodes.

    Paramaters:
    -----------
    x : torch.Tensor
        A tensor of shape (N, 3) representing a point cloud, where N is the
        number of points in the point cloud and B is the batch size.
    t : float
        The maximum distance between points for them to be connected by an
        edge.

    Returns:
    --------
    dgl.DGLGraph
        A graph where the nodes are the points in the point cloud and the edges
        are the connections between points that are closer than the threshold.
    """
    # calculate the distance between all pairs of points
    d = torch.cdist(x, x)

    # attach an edge between nodes if the distance is less than a threshold
    edges = torch.argwhere(d < t)

    # remove all self-connections
    edges = edges[edges[:, 0] != edges[:, 1]]

    # create a graph using the edges
    g = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=x.shape[0])
    g.ndata['coordinates'] = x
    g.edata['rel_pos'] = x[edges[:, 0]] - x[edges[:, 1]]

    return g