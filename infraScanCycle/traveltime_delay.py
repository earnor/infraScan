
import rasterio


def stack_tif_files(var):

    # List of your TIFF file paths
    tiff_files = [f"/s1_{var}.tif", f"/s2_{var}.tif", f"/s3_{var}.tif"]

    # Open the first file to retrieve the metadata
    with rasterio.open(r"data/independent_variable/processed/scenario"+tiff_files[0]) as src0:
        meta = src0.meta

    # Update metadata to reflect the number of layers
    meta.update(count=len(tiff_files))

    out_fp = fr"data/independent_variable/processed/scenario/scen_{var}.tif"
    # Read each layer and write it to stack
    with rasterio.open(out_fp, 'w', **meta) as dst:
        for id, layer in enumerate(tiff_files, start=1):
            with rasterio.open(r"data/independent_variable/processed/scenario"+layer) as src1:
                dst.write_band(id, src1.read(1))

