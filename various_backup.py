# test0 further densification for all keyframes in the list before mapping
for i in range(len(keyframe_list)):
    iter_time_idx = keyframe_list[i]['id']
    iter_color = keyframe_list[i]['color']
    iter_depth = keyframe_list[i]['depth']
    iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
    iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
            'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
    iter_data['semantic_id'] = keyframe_list[i]['semantic_id']
    iter_data['semantic_color'] = keyframe_list[i]['semantic_color']

    params, variables = add_new_gaussians(params, params_opt_exclude, variables, iter_data, 
                                    config['mapping']['sil_thres'], iter_time_idx, config['mean_sq_dist_method'],
                                    device, load_semantics=load_semantics)
    
# end test0