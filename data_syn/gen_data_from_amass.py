'''
Generate synthesis IMU data from AMASS
'''
import glob
import os
import threading

import cv2
import sys
# TODO
import torch
from tqdm import tqdm
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import numpy as np
import chumpy as ch
import pickle as pkl

# comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
comp_device = "cpu"
MODEL_PATH = 'data_syn/smplh/models/%s/model.npz'
male_fname = MODEL_PATH % 'male'
female_fname = MODEL_PATH % 'female'
num_betas = 16
model_male = BodyModel(bm_fname=male_fname, num_betas=num_betas).to(comp_device)
model_female = BodyModel(bm_fname=female_fname, num_betas=num_betas).to(comp_device)
# TODO
# Please modify here to specify which vertices to use
# 所有关节点
VERTEX_IDS = [3485, 914, 4402, 3023, 1047, 4531, 3017, 3436, 6731, 3506, 3361, 6759, 3061, 668, 4156, 3071, 1830, 6471,
              1667, 5136, 2210, 5671, 2001, 5459]

# TODO
TARGET_FPS = 60

# TODO
# 排除手和脚的关节点，胯部关节点使用采集到的
SMPL_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


def get_ori_acc_trans(A_global_list, joint_location, frame_rate, n):
    orientation = []  # 旋转矩阵
    acceleration = []  # 加速度
    for j in range(A_global_list.shape[0]):
        ori_tmp = []
        for i in range(24):
            a_global = A_global_list[j]
            ori_ = a_global[i, :3, :3]
            ori_tmp.append(ori_)

        orientation.append(np.array(ori_tmp))

    time_interval = 1.0 / frame_rate
    total_number = len(A_global_list)
    for idx in range(n, total_number - n):
        vertex_0 = joint_location[idx - n]  # 24 * 3
        vertex_1 = joint_location[idx]
        vertex_2 = joint_location[idx + n]
        # 1 加速度合成
        accel_tmp = (vertex_2 + vertex_0 - 2 * vertex_1) / (n * n * time_interval * time_interval)
        acceleration.append(accel_tmp)

    return np.array(orientation[n:-n]), np.array(acceleration)

def compute_imu_data(gender, betas, poses, trans, frame_rate, n):
    if gender == 'male':
        model = model_male
    else:
        model = model_female
    poses = np.array(poses)
    time_length = len(poses)

    body_parms = {
        'root_orient': torch.Tensor(poses[:, :3]).to(comp_device),  # controls the global root orientation
        'pose_body': torch.Tensor(poses[:, 3:66]).to(comp_device),  # controls the body
        'pose_hand': torch.Tensor(poses[:, 66:]).to(comp_device),  # controls the finger articulation
        'trans': torch.Tensor(trans).to(comp_device),  # controls the global body position
        'betas': torch.Tensor(np.repeat(betas[np.newaxis, :], repeats=time_length, axis=0)).to(comp_device),
        # controls the body shape. Body shape is static
    }

    body_pose_beta = model(
        **{k: v for k, v in body_parms.items() if k in ['pose_body', 'betas', 'root_orient', 'trans', 'pose_hand']})

    return body_pose_beta.A[:, SMPL_IDS, :], body_pose_beta.v[:, VERTEX_IDS, :], body_pose_beta.Jtr[:, SMPL_IDS, :]


def findNearest(t, t_list):
    list_tmp = np.array(t_list) - t
    list_tmp = np.abs(list_tmp)
    index = np.argsort(list_tmp)[:2]
    return index


# Turn MoCap data into 60FPS
def interpolation_integer(poses_ori, fps):
    poses = []
    n_tmp = int(fps / TARGET_FPS)
    poses_ori = poses_ori[::n_tmp]

    for t in poses_ori:
        poses.append(t)

    return poses


def interpolation(poses_ori, fps):
    poses = []
    total_time = len(poses_ori) / fps
    times_ori = np.arange(0, total_time, 1.0 / fps)
    times = np.arange(0, total_time, 1.0 / TARGET_FPS)

    for t in times:
        index = findNearest(t, times_ori)
        if len(index) != 2:
            continue
        if index[0] == len(poses_ori): break
        a = poses_ori[index[0]]
        t_a = times_ori[index[0]]
        if index[1] == len(poses_ori): break
        b = poses_ori[index[1]]
        t_b = times_ori[index[1]]

        if t_a == t:
            tmp_pose = a
        elif t_b == t:
            tmp_pose = b
        else:
            tmp_pose = a + (b - a) * ((t_b - t) / (t_b - t_a))
        poses.append(tmp_pose)

    return poses


# Extract pose parameter from pkl_path, save to res_path
def generate_data1(npz_dir, res_path):
    if os.path.exists(res_path):
        return
    
    data_in = np.load(npz_dir)
    
    if len(data_in['poses']) == 1:
        return

    data_out = {}
    data_out['gender'] = data_in['gender']
    data_out['betas'] = data_in['betas']

    # In case the original frame rates (eg 40FPS) are different from target rates (60FPS) 
    fps_ori = data_in['mocap_framerate']
    # print("frame rate = %s" % fps_ori)
    n = 4  # 加速度合成间隔帧数

    if (fps_ori % TARGET_FPS) == 0:
        data_out['poses'] = interpolation_integer(data_in['poses'], fps_ori)
        trans = interpolation_integer(data_in['trans'], fps_ori)
    else:
        data_out['poses'] = interpolation(data_in['poses'], fps_ori)
        trans = interpolation(data_in['trans'], fps_ori)

    data_out['betas'] = np.array(data_out['betas'])
    trans = np.array(trans)

    A, V, J = compute_imu_data(data_out['gender'], data_out['betas'], data_out['poses'], trans, TARGET_FPS, n)

    data_out_dict = {"A": A.cpu().numpy(), "V":V.cpu().numpy(), "J":J.cpu().numpy()}
    with open(res_path, 'wb') as fout:
        pkl.dump(data_out_dict, fout)

    # with open(res_path, 'rb') as fout:
    #     data_out_dict = pkl.load(fout)

    # A_global_arr, VERTEXS, joint_location = data_out_dict['A'], data_out_dict['V'], data_out_dict['J']
    # print(VERTEXS.shape)
    # mu = 0.008  # 单位米，移动小于mu则为支撑脚
    # s_foot = []
    # for idx in range(1, len(joint_location)):
    #     S_foot = [0, 0]
    #     if np.linalg.norm((joint_location[idx][10] - joint_location[idx - 1][10])) < mu:
    #         S_foot[0] = 1
    #     if np.linalg.norm((joint_location[idx][11] - joint_location[idx - 1][11])) < mu:
    #         S_foot[1] = 1

    #     s_foot.append(S_foot)
    # s_foot = np.array(s_foot)[n: -n + 1]  # 脚触底数据
    # position = VERTEXS[n:-n]  # 24个关节点在空间上的坐标 (x, y, z)

    # orientation, acceleration = get_ori_acc_trans(A_global_arr, VERTEXS, TARGET_FPS,
    #                                               n)  # orientation传感器方向（自身坐标系）, acceleration传感器加速度(自身坐标系)

    # print(s_foot.shape, position.shape, orientation.shape, acceleration.shape)

def generate_data2(npz_dir, res_path):
    if not os.path.exists(res_path):
        return
    if os.path.exists(res_path[:-8] + "_syn.pkl"):return

    data_in = np.load(npz_dir)
    if len(data_in['poses']) == 1:
        return

    data_out = {}
    data_out['gender'] = data_in['gender']
    data_out['betas'] = data_in['betas']

    # In case the original frame rates (eg 40FPS) are different from target rates (60FPS) 
    fps_ori = data_in['mocap_framerate']
    # print("frame rate = %s" % fps_ori)
    n = 4  # 加速度合成间隔帧数

    if (fps_ori % TARGET_FPS) == 0:
        data_out['poses'] = interpolation_integer(data_in['poses'], fps_ori)
        trans = interpolation_integer(data_in['trans'], fps_ori)
    else:
        data_out['poses'] = interpolation(data_in['poses'], fps_ori)
        trans = interpolation(data_in['trans'], fps_ori)

    data_out['betas'] = np.array(data_out['betas'])
    trans = np.array(trans)
    data_out['trans'] = trans[n:-n]
    # A, V, J = compute_imu_data(data_out['gender'], data_out['betas'], data_out['poses'], trans, TARGET_FPS, n)

    # data_out_dict = {"A": A.cpu().numpy(), "V":V.cpu().numpy(), "J":J.cpu().numpy()}
    # with open(res_path, 'wb') as fout:
    #     pkl.dump(data_out_dict, fout)

    with open(res_path, 'rb') as fout:
        data_out_dict = pkl.load(fout)

    A_global_arr, VERTEXS, joint_location = data_out_dict['A'], data_out_dict['V'], data_out_dict['J']
    mu = 0.008  # 单位米，移动小于mu则为支撑脚
    s_foot = []
    for idx in range(1, len(joint_location)):
        S_foot = [0, 0]
        if np.linalg.norm((joint_location[idx][10] - joint_location[idx - 1][10])) < mu:
            S_foot[0] = 1
        if np.linalg.norm((joint_location[idx][11] - joint_location[idx - 1][11])) < mu:
            S_foot[1] = 1

        s_foot.append(S_foot)
    s_foot = np.array(s_foot)[n: -n + 1]  # 脚触底数据
    position = VERTEXS[n:-n]  # 24个关节点在空间上的坐标 (x, y, z)

    orientation, acceleration = get_ori_acc_trans(A_global_arr, VERTEXS, TARGET_FPS,
                                                  n)  # orientation传感器方向（自身坐标系）, acceleration传感器加速度(自身坐标系)

    data_out['ori'], data_out['acc'], data_out['point'], data_out['s_foot'] = orientation, acceleration, position, s_foot
    for fdx in range(0, len(data_out['poses'])):
        pose_tmp = []  # np.zeros(0)
        for jdx in SMPL_IDS:
            tmp = data_out['poses'][fdx][jdx * 3:(jdx + 1) * 3]
            tmp = cv2.Rodrigues(tmp)[0].flatten().tolist()
            pose_tmp = pose_tmp + tmp

        data_out['poses'][fdx] = []
        data_out['poses'][fdx] = pose_tmp
    poses = np.array(data_out['poses'])

    data_out['poses'] = poses[n:-n]

    with open(res_path[:-8] + "_syn.pkl", 'wb') as fout:
        pkl.dump(data_out, fout)

    #print(res_path)



# Generate synthesic data for H3.6M
def main(pkl_path, res_data_path):
    generate_data2(pkl_path, res_data_path)


if __name__ == '__main__':
    from glob import glob

    files = glob("../acc2pos/dataset/*/*/*.npz")

    for file_ in tqdm(files):
        try:
            if "shape.npz" in file_:
                continue
            name_arr = file_.split("/")
            filename = name_arr[-1].split(".")[0]

            if filename == 'shape':
                continue
            new_file = "/".join(name_arr[:-1]) + "/" + filename + "_avj.pkl"

            pkl_path = file_
            res_data_path = new_file
            main(pkl_path, res_data_path)
            
        except Exception as e:
            print(e)
