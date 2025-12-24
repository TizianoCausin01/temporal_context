from matplotlib.colors import LinearSegmentedColormap
def truncate_colormap(cmap, minval=0.1, maxval=0.9, n=256):
    return LinearSegmentedColormap.from_list(
        f"trunc({cmap.name})",
        cmap(np.linspace(minval, maxval, n))
    )
