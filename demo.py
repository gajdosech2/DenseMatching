import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import sys
import cv2
sys.path.append("C:/GIT/DenseMatching/")

import argparse
import numpy as np
import time
import imageio
import matplotlib.pyplot as plt
import matplotlib.cm as cm    
from model_selection import model_type, pre_trained_model_types, select_model
from datasets.util import pad_to_same_shape
from utils_flow.visualization_utils import make_sparse_matching_plot  
from models.inference_utils import estimate_mask
from utils_flow.flow_and_mapping_operations import convert_flow_to_mapping
from validation.utils import matches_from_flow
from admin.stats import DotDict 

import torch
torch.set_grad_enabled(False)


# config 
model = 'PDCNet'
pre_trained_model = 'megadepth'
path_to_pre_trained_models = 'C:/GIT/DenseMatching/pre_trained_models/' 
flipping_condition = False 
global_optim_iter = 3
local_optim_iter = 7 
multi_stage_type = 'h'
confidence_map_R = 1.0
ransac_thresh = 1.0
mask_type = 'proba_interval_1_above_10'
homography_visibility_mask = True
scaling_factors = [0.5, 0.6, 0.88, 1, 1.33, 1.66, 2]
compute_cyclic_consistency_error = True 
uncertainty_key = 'p_r'
scale = 1

args = DotDict({'network_type': model, 'multi_stage_type': multi_stage_type, 'confidence_map_R': confidence_map_R, 
                'ransac_thresh': ransac_thresh, 'mask_type': mask_type, 
                'homography_visibility_mask': homography_visibility_mask, 'scaling_factors': scaling_factors, 
                'compute_cyclic_consistency_error': compute_cyclic_consistency_error})


def main(f1, f2, k_top=10000, skip=2, scale_factor=1.0):
    # load images
    query_img = cv2.imread(f1, cv2.IMREAD_UNCHANGED)
    width = int(query_img.shape[1] * scale_factor)
    height = int(query_img.shape[0] * scale_factor)
    dim = (width, height)
    query_img = cv2.resize(query_img, dim, interpolation = cv2.INTER_AREA)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    norm = np.zeros(query_img.shape)
    query_image = cv2.normalize(query_img,  norm, 0, 255*scale, cv2.NORM_MINMAX)

    reference_img = cv2.imread(f2, cv2.IMREAD_UNCHANGED)
    reference_img = cv2.resize(reference_img, dim, interpolation = cv2.INTER_AREA)
    reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
    norm = np.zeros(reference_img.shape)
    reference_image = cv2.normalize(reference_img,  norm, 0, 255*scale, cv2.NORM_MINMAX)

    # query_image = imageio.imread(sys.argv[1], pilmode='RGB')
    # reference_image = imageio.imread(sys.argv[2], pilmode='RGB')

    query_image_shape = query_image.shape
    ref_image_shape = reference_image.shape

    # cv2.imshow('Query', cv2.resize(query_image/255.0, (960, 540)))
    # cv2.waitKey(0)

    # cv2.imshow('Reference', cv2.resize(reference_image/255.0, (960, 540)))
    # cv2.waitKey(0)

    query_image_, reference_image_ = pad_to_same_shape(query_image, reference_image)
    query_image_ = torch.from_numpy(query_image_).permute(2, 0, 1).unsqueeze(0)
    reference_image_ = torch.from_numpy(reference_image_).permute(2, 0, 1).unsqueeze(0)

    network, estimate_uncertainty = select_model(
        model, pre_trained_model, args, global_optim_iter, local_optim_iter,
        path_to_pre_trained_models=path_to_pre_trained_models)
    estimate_uncertainty = True  

    print(query_image.shape)
    print(reference_image.shape)

    start = time.time()
    # run network 
    estimated_flow, uncertainty_components = network.estimate_flow_and_confidence_map(query_image_, reference_image_)
    print("Network Execution Time: " + str(int((time.time() - start) * 1000)) + " ms")
      
    # filter correspondences  
    confidence_map = uncertainty_components[uncertainty_key]
    confidence_map = confidence_map[:, :, :ref_image_shape[0], :ref_image_shape[1]]

    mask_padded = estimate_mask(mask_type, uncertainty_components) 
    if 'warping_mask' in list(uncertainty_components.keys()):
        mask_padded = uncertainty_components['warping_mask'] * mask_padded

    mask = mask_padded[:, :ref_image_shape[0], :ref_image_shape[1]]
    mapping_estimated = convert_flow_to_mapping(estimated_flow)
    mask = mask & mapping_estimated[:, 0].ge(0) & mapping_estimated[:, 1].ge(0) & mapping_estimated[:, 0].le(query_image_shape[1] - 1) & mapping_estimated[:, 1].le(query_image_shape[0] - 1)

    mkpts_query, mkpts_ref = matches_from_flow(estimated_flow, mask)

    confidence_values = confidence_map.squeeze()[mask.squeeze()].cpu().numpy()
    sort_index = np.argsort(np.array(confidence_values)).tolist()[::-1]
    confidence_values = np.array(confidence_values)[sort_index]
    mkpts_query = np.array(mkpts_query)[sort_index]
    mkpts_ref = np.array(mkpts_ref)[sort_index]
        
    # save top matches 
    mkpts_q = mkpts_query[:k_top:skip] / scale_factor
    mkpts_r = mkpts_ref[:k_top:skip] / scale_factor
    confidence_values = confidence_values[:k_top:skip]

    np.savetxt('ptsq.csv', np.asarray(mkpts_q), delimiter=',', fmt='%d')
    np.savetxt('ptsr.csv', np.asarray(mkpts_r), delimiter=',', fmt='%d')
    np.savetxt('conf.csv', np.asarray(confidence_values), delimiter=',', fmt='.3%f')

    color = cm.jet(confidence_values)
    out = make_sparse_matching_plot(query_image, reference_image, mkpts_q, mkpts_r, color, margin=10)
    plt.figure(figsize=(20,10))
    plt.imshow(out)
    plt.savefig("match.png", bbox_inches='tight')
    return str(len(mkpts_q))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Wrong number of input parameters")
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])     
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]))      
    elif len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))  
    elif len(sys.argv) == 6:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))  