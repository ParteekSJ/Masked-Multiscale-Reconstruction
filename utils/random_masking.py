import torch


def random_masking(x, mask_ratio, ids_shuffle=None):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    N - batch size, L - number of patches, D - patch dimensionality.
    """

    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))  # amount of patches to preserve

    # sort noise for each sample
    if ids_shuffle is None:
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


if __name__ == "__main__":
    x = torch.randn(1, 5, 2)
    x_masked, mask, ids_restore = random_masking(
        x,
        mask_ratio=0.4,
        noise=torch.tensor([[0.2, -0.5, 0.7, -1.2, 0.3]]),
    )


"""
noise(1x5 Tensor) = [[0.2, -0.5, 0.7, -1.2, 0.3]]
ids_shuffle: sorting + idx replace: [-1.2, -0.5, 0.2, 0.3, 0.7] => [3, 1, 0, 4, 2]
ids_restore = sorting + idx replace: [0, 1, 2, 3, 4] => [2, 1, 4, 0, 3]
ids_keep = ids_shuffle[:, :len_keep]: [3, 1, 0, 4, 2] => [3, 1, 0]

x_masked = Shape:[1, 3, 2]. noise with indices [3, 1, 0] is used.
mask = [1., 1., 1., 1., 1.]
mask[:, :len_keep] = 0 => [0., 0., 0., 1., 1.]
mask = torch.gather(mask, dim=1, index=ids_restore) => [0., 0., 1., 0., 1.]
"""
