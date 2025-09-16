"""
Copyright Zexian Zeng's lab, AAIS, Peking Universit. All Rights Reserved

@author: Yufeng He
"""

import os

import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Union
from joblib import Parallel, delayed
from skimage import io
import pandas as pd
import tifffile as tiff

from .anchor import set_seed, ModelGenerator, RegionExtractor, AnchorGenerator
from .finetune import gradient_map, create_kernel, refine_mask_parallel

os.environ["IOPATH_DISABLE_TELEMETRY"] = "1"

def parallel_gm(img_cell, cell_box, index, gene_mtx, kernel, alpha, expand_by, gene, neighbor):
    non_zero_positions = gradient_map(img_cell=img_cell, cell_box=cell_box, index=index, gene_mtx=gene_mtx, kernel=kernel, alpha=alpha, expand_by=expand_by, gene=gene, neighbor=neighbor)
    return non_zero_positions

np.seterr(invalid='ignore', divide='ignore')
def segmentation(
    img_path, platform, gene_mtx_filename, config_file, weights_file, output="./results", alpha=0.5, expand_by=5, gene=True, random_seed=2024, n_jobs=16, 
    num_proposals=None, sample_step=None, renewal_thres=None, isslice=False, x0=None, y0=None, length=None, width=None, threshold=-0.20, minarea=600, 
    maxarea=1200, fov=None, stride=128, ratio_stride=3, neighbor=8
):

    set_seed(random_seed)
    model = ModelGenerator(config_file, weights_file, num_proposals, sample_step, renewal_thres).model

    st_data = STReader(platform, img_path, gene_mtx_filename, fov=fov)

    img_test = st_data.img
    df_test = st_data.df
    # assert (img_test.shape[0] >= (df_test['y'].max() - 5)) & (img_test.shape[0] <= (df_test['y'].max() + 5)) 
    # assert (img_test.shape[1] >= (df_test['x'].max() - 5)) & (img_test.shape[1] <= (df_test['x'].max() + 5))
    
    extractor = RegionExtractor(img=img_test, gene_df=df_test)
    
    if isslice == False:
        x0, y0 = 0, 0
        length = img_test.shape[0]
        width = img_test.shape[1]
    else:
        assert (x0 != None) & (y0 != None) & (length != None) & (x0 != None)
        
    img_test, gene_df_test = extractor.slice_region(
        x0, y0, length, width
    )

    anchors = AnchorGenerator(
        img_cell=img_test, model=model, threshold=threshold, batch_size=4, minarea=minarea, maxarea=maxarea
    )
    boxes_test, scores = anchors.process(stride=stride, ratio_stride=ratio_stride, n_jobs=n_jobs, output=output)
    scores = scores.numpy()
    kernel = create_kernel(3)

    gene_df_test["gene_idx"] = gene_df_test['geneID'].map({gene_name: idx for idx, gene_name in enumerate(set(gene_df_test['geneID']))})
    gene_sparse_test = gene_df_test[["gene_idx", "x", "y", "MIDCount"]].values.astype(np.int32)
    sorted_indices_test = np.argsort(scores)[::-1].copy()
    boxes_sorted_test = boxes_test[sorted_indices_test]
    results = Parallel(n_jobs=n_jobs)(delayed(parallel_gm)(img_test, box, idx, gene_sparse_test, kernel, alpha, expand_by, gene, neighbor) for idx, box in tqdm(enumerate(boxes_sorted_test), total=boxes_sorted_test.shape[0], desc="Processing boxes"))
    mask = np.zeros_like(img_test, dtype=np.int32)
    for value, position in tqdm(enumerate(results)):
        if position[0] is not None and len(position[0].shape) == 2:
            for y, x in position[0]:
                mask[y, x] = value
    np.save(os.path.join(output, "mask.npy"), mask)
    return mask
    
class STReader:
    def __init__(
        self,
        platform: str,
        img_path: Union[str, Path],
        gene_mtx_filename: Union[str, Path],
        fov: int = None
    ):
        valid_platforms = ['stereoseq', 'xenium', 'nanostring', 'hd']
        if platform not in valid_platforms:
            raise ValueError(f"Invalid platform '{platform}'. Must be one of {valid_platforms}.")

        self.platform = platform
        self.img_path = img_path
        self.gene_mtx_filename = gene_mtx_filename
        self.fov = fov
        self.img, self.df = self.load_st()

    def load_st(self) -> Tuple[np.ndarray, pd.DataFrame]:

        if type(self.img_path) == np.ndarray:
            img = self.img_path
        else:
            try:
                img = io.imread(self.img_path)
                img = img / np.max(img) * 256
                if len(img.shape) == 3:
                    img = img[:, :, 0]
            except Exception as e:
                try:
                    img = tiff.TiffReader(self.img_path).pages[0].asarray()
                except Exception as e:    
                    raise RuntimeError(
                        f"Error occurred while reading the image from {self.img_path}: {str(e)}"
                    )
        if type(self.gene_mtx_filename) == pd.core.frame.DataFrame:
            df = self.gene_mtx_filename
        else:
            if self.platform == 'stereoseq':
                df = self._load_gene_mtx_stereoseq()
            elif self.platform == 'xenium':
                df = self._load_gene_mtx_xenium()
            elif self.platform == 'nanostring':
                df = self._load_gene_mtx_nanostring()
            
        print("Successfuly read in your ST data.")

        return img, df

    def _load_gene_mtx_stereoseq(self) -> pd.DataFrame:
        try:
            if self.gene_mtx_filename.endswith("gz"):
                try: 
                    df = pd.read_csv(self.gene_mtx_filename, sep="\t", compression="gzip")
                except Exception as e:
                    df = pd.read_csv(self.gene_mtx_filename, sep="\t", compression="gzip", skiprows=6)
            else:
                try: 
                    df = pd.read_csv(self.gene_mtx_filename, sep="\t")
                except Exception as e:
                    df = pd.read_csv(self.gene_mtx_filename, sep="\t", skiprows=6)
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while loading the gene matrix from {self.gene_mtx_filename}: {str(e)}"
            )
        assert 'geneID' in df.columns, "No geneID in the columns of gene csv file"
        assert 'x' in df.columns, "No x in the columns of gene csv file"
        assert 'y' in df.columns, "No y in the columns of gene csv file"
        assert 'MIDCount' in df.columns, "No MIDCount in the columns of gene csv file"
        
        return df

    def _load_gene_mtx_xenium(self) -> pd.DataFrame:
        try:
            df = pd.read_parquet(self.gene_mtx_filename)
        except Exception as e:
            try:
                df = pd.read_csv(self.gene_mtx_filename)
            except Exception as e:
                try:
                    df = pd.read_csv(self.gene_mtx_filename, sep="\t")
                except Exception as e:
                    raise RuntimeError(
                        f"Error occurred while loading the gene matrix from {self.gene_mtx_filename}: {str(e)}"
                    )
        assert 'feature_name' in df.columns, "No feature_name in the columns of gene csv file"
        assert 'x_location' in df.columns, "No x_location in the columns of gene csv file"
        assert 'y_location' in df.columns, "No y_location in the columns of gene csv file"
        df['x'] = (df['x_location'] / 0.2125).astype(np.int32)
        df['y'] = (df['y_location'] / 0.2125).astype(np.int32)
        df['geneID'] = df['feature_name']
        df.loc[:, "MIDCount"] = 1
        df = df[["geneID", 'x', 'y', "MIDCount"]]
        return df
    
    def _load_gene_mtx_nanostring(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.gene_mtx_filename)
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while loading the gene matrix from {self.gene_mtx_filename}: {str(e)}"
            )
        try:
            num_fov = len(np.unique(df['fov']))
        except Exception as e:
            raise RuntimeError("Confirm your NanoString gene csv file (No fov column)")  
            
        assert 'fov' in df.columns, "No fov in the columns of gene csv file"
        assert 'x_local_px' in df.columns, "No x_local_px in the columns of gene csv file"
        assert 'y_local_px' in df.columns, "No y_local_px in the columns of gene csv file"
        assert 'target' in df.columns, "No target in the columns of gene csv file"  
        
        if num_fov > 1:
            assert self.fov != None, f"Only one fov should be given, choose from {np.unique(df['fov'])}."
            df['x'], df['y'], df['gene_ID'] = df['x_local_px'], df['y_local_px'], df['target']
            df.loc[:, "MIDCount"] = 1
            df = df[df['fov'] == self.fov]
            assert df.shape[0] >= 1, "No transcripts found in this fov"
            df = df[["gene_ID", 'x', 'y', "MIDCount"]]
        else:
            df['x'], df['y'], df['gene_ID'] = df['x_local_px'], df['y_local_px'], df['target']
            df.loc[:, "MIDCount"] = 1
            df = df[["gene_ID", 'x', 'y', "MIDCount"]]
        return df
    
    def generate_spot_matrix(self) -> np.ndarray:
        """
        Generate spot matrix from a dataframe and image.
        """
        # Convert the x and y columns to numpy arrays
        x_values = self.df["x"].values
        y_values = self.df["y"].values

        # Create an empty matrix of the specified dimensions
        spot_matrix = np.zeros(self.img.shape)

        # Fill the matrix with the corresponding MIDCount values
        for x, y, MIDCount in zip(x_values, y_values, self.df["MIDCount"]):
            spot_matrix[y, x] += MIDCount

        return spot_matrix

