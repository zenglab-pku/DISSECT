"""
This module is used to read BGI data and image file, and return an AnnData object.
"""
import itertools
import pandas as pd
import numpy as np
import polars as pl
from tifffile import imread
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from skimage import io
from scipy.sparse import csr_matrix # type: ignore
from anndata import AnnData

import cv2


def load_bin(
    gem_file: str,
    image_file: str,
    bin_size: int,
    library_id: str,
) -> AnnData:
    """
    Read BGI data and image file, and return an AnnData object.
    Parameters
    ----------
    gem_file
        The path of the BGI data file.
    image_file
        The path of the image file.
    bin_size
        The size of the bin.
    library_id
        The library id.
    Returns
    -------
    Annotated data object with the following keys:

        - :attr:`anndata.AnnData.obsm` ``['spatial']`` - spatial spot coordinates.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{library_id}']['images']`` - *hires* images.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{library_id}']['scalefactors']`` - scale factors for the spots.
    """ # noqa: E501
    library = library_id
    dat_file = gem_file
    image = image_file
    bin_s = bin_size
    ###########################
    # different gem have different delimiter!!!!!!!
    # COAD: " " , other may be "\t"
    dat = pd.read_csv(dat_file, delimiter="\t", comment="#")
    if image.endswith("tiff"):
        image = imread(image)
    else:
        image = cv2.imread(image)
    ######
    dat['x'] -= dat['x'].min()
    dat['y'] -= dat['y'].min()

    width = dat['x'].max() + 1
    height = dat['y'].max() + 1
    ###
    dat['xp'] = (dat['x'] // bin_s) * bin_s
    dat['yp'] = (dat['y'] // bin_s) * bin_s
    dat['xb'] = np.floor(dat['xp'] / bin_s + 1).astype(int)
    dat['yb'] = np.floor(dat['yp'] / bin_s + 1).astype(int)

    dat['bin_ID'] = max(dat['xb']) * (dat['yb'] - 1) + dat['xb']
    ###
    trans_x_xb = dat[['x', 'xb']].drop_duplicates()
    trans_x_xb = trans_x_xb.groupby('xb')['x'].apply(
        lambda x: int(np.floor(np.mean(x)))).reset_index()
    trans_y_yb = dat[['y', 'yb']].drop_duplicates()
    trans_y_yb = trans_y_yb.groupby('yb')['y'].apply(
        lambda y: int(np.floor(np.mean(y)))).reset_index()

    trans_matrix = pd.DataFrame(list(itertools.product(
        trans_x_xb['xb'], trans_y_yb['yb'])), columns=['xb', 'yb'])
    trans_matrix = pd.merge(trans_matrix, trans_x_xb, on='xb')
    trans_matrix = pd.merge(trans_matrix, trans_y_yb, on='yb')
    trans_matrix['bin_ID'] = max(
        trans_matrix['xb']) * (trans_matrix['yb'] - 1) + trans_matrix['xb']

    trans_matrix['in_tissue'] = 1

    tissue_positions = pd.DataFrame()
    # barcode is str, not number
    tissue_positions['barcodes'] = trans_matrix['bin_ID'].astype(str)
    tissue_positions['in_tissue'] = trans_matrix['in_tissue']
    tissue_positions['array_row'] = trans_matrix['yb']
    tissue_positions['array_col'] = trans_matrix['xb']
    tissue_positions['pxl_row_in_fullres'] = trans_matrix['y']
    tissue_positions['pxl_col_in_fullres'] = trans_matrix['x']
    tissue_positions.set_index('barcodes', inplace=True)

    ###
    if 'MIDCount' in dat.columns:
        dat = dat.groupby(['geneID', 'xb', 'yb'])[
            'MIDCount'].sum().reset_index()
        dat['bin_ID'] = max(dat['xb']) * (dat['yb'] - 1) + dat['xb']

        ###
        unique_genes = dat['geneID'].unique()
        unique_barcodes = dat['bin_ID'].unique()
        gene_hash = {gene: index for index, gene in enumerate(unique_genes)}
        barcodes_hash = {barcodes: index for index,
                         barcodes in enumerate(unique_barcodes)}
        dat['gene'] = dat['geneID'].map(gene_hash)
        dat['barcodes'] = dat['bin_ID'].map(barcodes_hash)

        ###
        counts = csr_matrix((dat['MIDCount'], (dat['barcodes'], dat['gene'])))

    else:
        dat = dat.groupby(['geneID', 'xb', 'yb'])[
            'MIDCounts'].sum().reset_index()
        dat['bin_ID'] = max(dat['xb']) * (dat['yb'] - 1) + dat['xb']
        ###
        unique_genes = dat['geneID'].unique()
        unique_barcodes = dat['bin_ID'].unique()
        gene_hash = {gene: index for index, gene in enumerate(unique_genes)}
        barcodes_hash = {barcodes: index for index,
                         barcodes in enumerate(unique_barcodes)}
        dat['gene'] = dat['geneID'].map(gene_hash)
        dat['barcodes'] = dat['bin_ID'].map(barcodes_hash)

        ###
        counts = csr_matrix((dat['MIDCounts'], (dat['barcodes'], dat['gene'])))
    adata = AnnData(counts)
    adata.var_names = list(gene_hash.keys())
    adata.obs_names = list(map(str, barcodes_hash.keys()))
    ##########
    adata.obs = adata.obs.join(tissue_positions, how="left")
    adata.obsm['spatial'] = adata.obs[[
        'pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
    adata.obs.drop(columns=['in_tissue', 'array_row', 'array_col',
                   'pxl_row_in_fullres', 'pxl_col_in_fullres'], inplace=True,)
    ###
    spatial_key = "spatial"
    adata.uns[spatial_key] = {library: {}}
    adata.uns[spatial_key][library]["images"] = {}
    adata.uns[spatial_key][library]["images"] = {"hires": image}
    # tissue image / RNA shape
    tissue_hires_scalef = max(image.shape[0]/width, image.shape[1]/height)

    # the diameter of detection area(the spot that contains tissue)
    # can be adjust out side by size= in scatter function
    spot_diameter = bin_s / tissue_hires_scalef
    
    #fiducial_area = max(tissue_positions['array_row'].max() - tissue_positions['array_row'].min(),
    #                    tissue_positions['array_col'].max() - tissue_positions['array_col'].min())
    adata.uns[spatial_key][library]["scalefactors"] = {
        "tissue_hires_scalef": tissue_hires_scalef,
        "spot_diameter_fullres": spot_diameter,
    }

    return adata


def load_cell(
    gem_file: str,
    image_file: str,
    mask_file: int,
    library_id: str,
) -> AnnData:
    """
    Read transcriptomic data and image file, and return an AnnData object.
    Parameters
    ----------
    gem_file
        The path of the BGI data file.
    image_file
        The path of the image file.
    bin_size
        The size of the bin.
    library_id
        The library id.
    Returns
    -------
    Annotated data object with the following keys:

        - :attr:`anndata.AnnData.obsm` ``['spatial']`` - spatial spot coordinates.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{library_id}']['images']`` - *hires* images.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{library_id}']['scalefactors']`` - scale factors for the spots.
    """ # noqa: E501
    print("start reading")
    mask = np.load(mask_file, allow_pickle=True)
    dat = pl.read_csv(gem_file, separator=",", comment_prefix="#")
    image = np.load(image_file)
    
    spatial_key = 'spatial'
    library = library_id
    
    nonzero_indices = np.nonzero(mask > 0)
    transposed_indices = np.transpose(nonzero_indices)
    nonzero_values = mask[nonzero_indices]
    tmp_mask_array = np.column_stack((transposed_indices, nonzero_values))
    
    mask = pl.DataFrame(tmp_mask_array, schema=['x', 'y', 'barcodes'])
    
    print("start merging mask and counts")
    # Merge operation
    mask_data = mask.join(dat, on=['x', 'y'], how='left')
    mask_data = mask_data.with_columns('geneID').fill_null("NAGENE")
    mask_data = mask_data.with_columns('MIDCount').fill_null(0)
    try:
        mask_data = mask_data.with_columns('ExonCount').fill_null(0)
    except:
        print("No ExonCount column")

    # Groupby and sum
    exp = mask_data.groupby(['geneID', 'barcodes']).agg(pl.col('MIDCount').sum())
    exp = pl.from_pandas(exp.to_pandas().reset_index())

    # Construct count matrix
    unique_genes = exp.select('geneID').unique().to_numpy().ravel()
    unique_barcodes = exp.select('barcodes').unique().to_numpy().ravel()
    gene_hash = {gene: index for index, gene in enumerate(unique_genes)}
    barcodes_hash = {barcode: index for index, barcode in enumerate(unique_barcodes)}
    
    # Convert the hash maps to Polars DataFrames
    gene_df = pl.DataFrame({'geneID': list(gene_hash.keys()), 'gene': list(gene_hash.values())})
    
    barcode_df = pl.DataFrame({'barcodes': list(barcodes_hash.keys()), 'barcode_index': list(barcodes_hash.values())})
    
    # Perform join operations
    exp = exp.join(gene_df, on='geneID', how='left').join(barcode_df, on='barcodes', how='left')
    
    """
    exp = exp.with_columns([
        exp['geneID'].apply(lambda x: gene_hash[x]).alias('gene'),
        exp['barcodes'].apply(lambda x: barcodes_hash[x]).alias('barcode_index')
    ])
    """

    # Convert to pandas for csr_matrix compatibility
    exp_pd = exp.to_pandas()
    counts = csr_matrix((exp_pd['MIDCount'], (exp_pd['barcode_index'], exp_pd['gene'])))
    
    print("start building anndata")
    adata = AnnData(counts)
    adata.var_names = list(gene_hash.keys())
    adata.obs_names = list(map(str, barcodes_hash.keys()))
    """
    mask = mask.with_columns([
        (pl.col('x') - mask['x'].min()).alias('x'),
        (pl.col('y') - mask['y'].min()).alias('y')
    ])
    """
    # Coordinate normalization


    # Groupby and mean
    grouped_mask = mask.groupby('barcodes').agg([
        pl.col('x').mean().alias('center_x'),
        pl.col('y').mean().alias('center_y')
    ]).to_pandas()  # Convert to pandas DataFrame

    # Set index and join
    grouped_mask.set_index('barcodes', inplace=True)
    adata.obs['cell_id'] = "cell_" + adata.obs.index
    grouped_mask.index = grouped_mask.index.astype(str)
    adata.obs = adata.obs.join(grouped_mask, how="left")
    adata.obsm['spatial'] = adata.obs[["center_x", "center_y"]].to_numpy()
    adata.obs.drop(columns=["center_x", "center_y"], inplace=True)
    
    adata.uns[spatial_key] = {library: {}}
    adata.uns[spatial_key][library]["images"] = {"hires": image}

    tissue_hires_scalef = 1 #max((mask['x'].max() + 1) / image.shape[1], (mask['y'].max() + 1) / image.shape[0])
    
    adata.uns[spatial_key][library]["scalefactors"] = {
        "tissue_hires_scalef": tissue_hires_scalef,
        "spot_diameter_fullres": 250,
    }
    return adata[:, adata.var.index != "NAGENE"]