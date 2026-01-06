import os
import mmengine
import mmcv
from pyquaternion import Quaternion

import cv2
import numpy as np
from mmengine.fileio import get
from tqdm import tqdm


def draw_bev_global_only(ego2global, cam2egos, sensor_names, trajectory, canvas_size=(448, 1200)):
    """
    绘制单侧鸟瞰图：展示自车全局行驶轨迹，并在当前帧位置标出所有相机的全局位姿
    """
    # 1. 初始化白色背景画布
    bev = np.ones((canvas_size[0], canvas_size[1], 3), dtype=np.uint8) * 255
    
    # 绘图参数
    center_x, center_y = canvas_size[1] // 2, canvas_size[0] // 2
    global_scale = 5.0  # 增加缩放比例以便看清相机位姿
    
    # 以第一帧位置为画布中心原点
    origin_pos = trajectory[0]

    # 2. 绘制历史轨迹 (灰色线)
    if len(trajectory) > 1:
        for i in range(1, len(trajectory)):
            p1 = trajectory[i-1] - origin_pos
            p2 = trajectory[i] - origin_pos
            pt1 = (int(center_x + p1[0] * global_scale), int(center_y - p1[1] * global_scale))
            pt2 = (int(center_x + p2[0] * global_scale), int(center_y - p2[1] * global_scale))
            cv2.line(bev, pt1, pt2, (200, 200, 200), 2, cv2.LINE_AA)

    # 3. 计算并绘制当前帧自车和相机的全局位姿
    # 自车当前全局位置
    curr_ego_pos = trajectory[-1] - origin_pos
    ego_pt = (int(center_x + curr_ego_pos[0] * global_scale), int(center_y - curr_ego_pos[1] * global_scale))
    
    # 定义相机颜色映射 (让不同相机更容易区分)
    colors = [
        (255, 0, 0),   # Front Left - 蓝色
        (0, 255, 0),   # Front - 绿色
        (0, 0, 255),   # Front Right - 红色
        (255, 165, 0), # Back Left - 橙色
        (0, 255, 255), # Back - 黄色
        (255, 0, 255)  # Back Right - 紫色
    ]

    for i, cam2ego in enumerate(cam2egos):
        # 核心转换：T_global = T_ego2global * T_cam2ego
        cam2global = ego2global @ cam2ego
        
        # 提取全局坐标 (x, y)
        c_pos_global = cam2global[:2, 3] - origin_pos
        c_px = int(center_x + c_pos_global[0] * global_scale)
        c_py = int(center_y - c_pos_global[1] * global_scale)
        
        # 提取全局朝向 (取相机 Z 轴在全局 X-Y 平面的投影)
        # cam2global[:2, 2] 是相机坐标系 Z 轴在全局系下的方向向量
        forward_vec = cam2global[:2, 2]
        
        # 计算箭头终点
        arrow_len = 25 
        end_px = int(c_px + forward_vec[0] * arrow_len)
        end_y = int(c_py - forward_vec[1] * arrow_len)
        
        # 绘制带箭头的线段代表相机位姿
        color = colors[i % len(colors)]
        cv2.arrowedLine(bev, (c_px, c_py), (end_px, end_y), color, 2, tipLength=0.3)
        
        # # 标注相机简称
        # short_name = sensor_names[i].replace('CAM_', '')
        # cv2.putText(bev, short_name, (c_px - 10, c_py - 10), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

    # 4. 绘制自车中心点
    cv2.circle(bev, ego_pt, 5, (0, 0, 0), -1)
    cv2.putText(bev, "GLOBAL BEV: Trajectory & Camera Poses", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return bev

data_path = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-mini'
anno_root = "/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_cam/mini/"
imageset = anno_root + "nuscenes_mini_infos_val_sweeps_occ.pkl"
output_dir = '/home/lianghao/wangyushen/data/wangyushen/Output/debug/scene_videos'
os.makedirs(output_dir, exist_ok=True)

if __name__ == '__main__':
    data = mmengine.load(imageset)
    scene_infos = data['infos']
    
    # 定义 2x3 拼接的布局
    sensor_layout = [
        ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
        ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    ]
    
    target_size = (400, 224)  # (width, height)
    fps = 12  # nuScenes 摄像头采样率大约为 12Hz

    for scene_token, scene_data in scene_infos.items():
        print(f'Processing Scene: {scene_token}')
        video_path = os.path.join(output_dir, f'{scene_token}.mp4')

       # 视频总尺寸：宽 1200，高 (224*2 + 448) = 896
        video_size = (target_size[0] * 3, target_size[1] * 2 + 448)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, video_size)

        key_frame_num = 0
        frame_idx = 0
        
        trajectory = [] 
        last_ego2global = None
        last_cam2egos = []
        last_sensor_names = []
        
        for info in tqdm(scene_data):
            # 第几帧
            frame_idx += 1
            
            # 是否是关键帧
            is_key_frame = info['is_key_frame']
            if is_key_frame:
                key_frame_num += 1 

            
            # --- 1. 更新或继承位姿数据 ---
            if is_key_frame: # todo 仅 is_key_frame=True的帧包含'LIDAR_TOP'信息
                # 关键帧：提取新位姿
                ego_pos = np.asarray(info['data']['LIDAR_TOP']['pose']['translation'])
                trajectory.append(ego_pos[:2])
                
                # 更新自车全局位姿
                curr_ego2global = np.eye(4)
                curr_ego2global[:3, :3] = Quaternion(info['data']['LIDAR_TOP']['pose']['rotation']).rotation_matrix
                curr_ego2global[:3, 3] = ego_pos
                last_ego2global = curr_ego2global          
                
                # 更新相机位姿
                current_cam2egos = []
                current_names = []
                for cam_type in [s for row in sensor_layout for s in row]:
                    c2e = np.eye(4) # todo 相机到自车的位置
                    c2e[:3, :3] = Quaternion(info['data'][cam_type]['calib']['rotation']).rotation_matrix
                    c2e[:3, 3] = np.asarray(info['data'][cam_type]['calib']['translation'])
                    current_cam2egos.append(c2e)
                    current_names.append(cam_type)
                
                last_cam2egos = current_cam2egos
                last_sensor_names = current_names            
            else:
                # 非关键帧：如果轨迹为空（第一帧不是关键帧的情况），初始化一个
                if len(trajectory) == 0:
                    # 尝试从当前帧抓取（如果sweep里有的话），否则跳过或设为原点
                    try:
                        ego_pos = np.asarray(info['data']['LIDAR_TOP']['pose']['translation'])
                        trajectory.append(ego_pos[:2])
                    except:
                        trajectory.append(np.array([0, 0]))
                else:
                    # 重点：非关键帧也要追加位置，否则轨迹图会跳变
                    # 如果 sweep 里有位姿，用 sweep 的；没有就继承上一个
                    try:
                        ego_pos = np.asarray(info['data']['LIDAR_TOP']['pose']['translation'])
                        trajectory.append(ego_pos[:2])
                        # 同时也更新一下 ego2global 保证旋转也是丝滑的
                        last_ego2global[:3, 3] = ego_pos
                    except:
                        trajectory.append(trajectory[-1])            
            
            
            # 获取每个相机相对于自车的位姿
            cam2egos = []
            sensor_names_flat = []
            # 获取每个相机视角的图像
            rows = []
            for row_sensors in sensor_layout:
                row_imgs = []
                for cam_type in row_sensors:
                    # 1. 获取路径并读取
                    img_path = os.path.join(data_path, info['data'][cam_type]['filename'])
                    img_byte = get(img_path, backend_args=None)
                    
                    # 2. 解码 (注意: cv2/mmcv 默认 BGR, 若 channel_order='rgb' 则为 RGB)
                    img = mmcv.imfrombytes(img_byte, flag='unchanged', backend='pillow', channel_order='rgb')
                    
                    # 3. 缩放
                    img_resized = mmcv.imresize(img, target_size)
                    
                    # 4. 转换回 BGR 以便 cv2.VideoWriter 写入 (如果是 RGB 读入的话)
                    img_bgr = img_resized
                    row_imgs.append(img_bgr)
                # 水平拼接一排
                rows.append(np.hstack(row_imgs))
            # 垂直拼接两排
            combined_cameras = np.vstack(rows)
            
            # 2. 生成 BEV 可视化图
            # 使用“最后一次有效”的数据，保证每一帧都有东西画，不会闪烁
            if last_ego2global is not None:
                bev_view = draw_bev_global_only(
                    last_ego2global, 
                    last_cam2egos, 
                    last_sensor_names, 
                    trajectory, 
                    canvas_size=(448, target_size[0] * 3)
                )
            else:
                # 万一第一帧没数据，画个白板
                bev_view = np.ones((448, target_size[0] * 3, 3), dtype=np.uint8) * 255

            # 3. 垂直合并相机图与 BEV 图
            final_frame = np.vstack([combined_cameras, bev_view])
            # 2. 在大图左上角标出帧信息
            # 准备文字内容
            if is_key_frame:
                text = f"Frame: {frame_idx}"
                color = (0, 0, 255) # RGB 下的红色
            else:
                text = f"Frame: {frame_idx}"
                color = (0, 0, 0)   # 黑色

            # 绘制背景框（防止黑色文字在深色背景看不清，可选）
            cv2.rectangle(final_frame, (5, 5), (550, 60), (255, 255, 255), -1)
            # 绘制文字
            cv2.putText(
                final_frame, 
                text, 
                (30, 50),                   # 坐标
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5,                        # 字体大小
                color, 
                2,                          # 粗细
                cv2.LINE_AA
            )            
            # 写入视频帧
            video_writer.write(final_frame)

        video_writer.release()
        print(f'Video saved to: {video_path}')
        print(f'scene token {scene_token} key frames: {key_frame_num}')