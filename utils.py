from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd

def read_tiff(path):
    """
    path - Path to the multipage-tiff file
    """
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    return np.array(images)

def load_image(df_features, transform, datapath1, datapath2):
    transformed_images = []
    for path in df_features['rescaled_2D_single_cell_tiff_path']:
        filename = os.path.basename(path)
        local_image_path = os.path.join(datapath1, filename)
        if not os.path.exists(local_image_path):
              local_image_path = os.path.join(datapath2, filename)
        if local_image_path:
            if os.path.exists(local_image_path):  
                images = read_tiff(local_image_path)
                image = images[1]
                
                # Resize the image to the desired dimensions using PIL
                image_copy = image.copy()
                image_resized = cv2.resize(image_copy, (224,224)) #uint8
                image_rgb = np.repeat(np.expand_dims(image_resized, axis=2), repeats=3, axis=2)

                transformed_image = transform(image_rgb)
                transformed_images.append(transformed_image)

    return transformed_images

def process_csv(csvpath1, csvpath2):
    df_feats_1 = pd.read_csv(csvpath1)
    df_feats_2 = pd.read_csv(csvpath2)

    # Dataset 1 had 0 scores for cells to indicate absence of structure/gfp
    df_feats_1["gfp_keep"] = (df_feats_1.kg_structure_org_score > 0) & (df_feats_1.mh_structure_org_score > 0)

    # Dataset 2 scored afterword; 0 in "no_structure" indicates absence of structure/gfp in cell; all other cells are NaN
    df_feats_2["gfp_keep"] = df_feats_2.no_structure.isna()

    all_fish_df = pd.concat(
            [df_feats_1, df_feats_2]
        ).reset_index(drop=True)

    all_fish_df.napariCell_ObjectNumber = all_fish_df.napariCell_ObjectNumber.astype(int).astype(str)
    all_fish_df["Type"] = "FISH"
    return all_fish_df

def process_meta(metapath1, metapath2):
    df_gs_1 = pd.read_csv(metapath1)
    df_gs_2 = pd.read_csv(metapath2)

    all_gs_df = pd.concat(
            [df_gs_1, df_gs_2]
        ).reset_index(drop=True)

    all_gs_df.napariCell_ObjectNumber = all_gs_df.napariCell_ObjectNumber.astype(int).astype(str)

    all_gs_df = all_gs_df.rename(
        columns={"original_fov_location": "fov_path"}
    )
    all_gs_df = all_gs_df.drop(
        columns=["Age", "Dataset"]
    )
    
    return all_gs_df
