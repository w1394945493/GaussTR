import os
import torch
import numpy as np
from numpy import random
import mmcv
from PIL import Image
import math
from copy import deepcopy
import torch.nn.functional as F
import cv2
from mmengine.fileio import get
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __init__(self, keys = ['img','output_img']): #!
        self.keys = keys
        return

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        for key in self.keys:
            if key in results:
                if isinstance(results[key], list):
                    # process multiple imgs in single frame
                    imgs = [img.transpose(2, 0, 1) for img in results[key]]
                    imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                else:
                    imgs = np.ascontiguousarray(results[key].transpose(2, 0, 1))
                results[key] = torch.from_numpy(imgs)
        return results

    def __repr__(self):
        return self.__class__.__name__


@TRANSFORMS.register_module()
class NuScenesAdaptor(object):
    def __init__(self, num_cams, use_ego=False):
        self.num_cams = num_cams
        self.projection_key = 'ego2img' if use_ego else 'lidar2img' # todo 'lidar2img'

    def __call__(self, input_dict):
        input_dict["projection_mat"] = np.float32(
            np.stack(input_dict[self.projection_key]) 
        )
        input_dict["image_wh"] = np.ascontiguousarray(
            np.array(input_dict["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
        )
        return input_dict


@TRANSFORMS.register_module()
class ResizeCropFlipImage(object):
    def __call__(self, results):
        aug_configs = results.get("aug_configs")
        resize, resize_dims, crop, flip, rotate, output_resize, output_dims, output_crop = aug_configs
        
        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        transforms = []
        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            #!----------------------------------------------------#
            mat = np.eye(4)
            mat[:3, :3] = ida_mat # todo 
            new_imgs.append(np.array(img).astype(np.float32))
            # todo (wys 12.30)
            transforms.append(mat) # todo img2img' 矩阵
            results["lidar2img"][i] = mat @ results["lidar2img"][i] # todo 已经都做过变换了
            results["ego2img"][i] = mat @ results["ego2img"][i]

        
        results["img"] = new_imgs
        results["img_shape"] = [x.shape[:2] for x in new_imgs]
        # todo (wys 12.30)
        results["img_aug_mat"] = np.array(transforms).astype(np.float32)
        
        if 'output_img' in results:
            imgs = results["output_img"]
            N = len(imgs)
            new_imgs = []
            transforms = []
            for i in range(N):
                img = Image.fromarray(np.uint8(imgs[i]))
                img, ida_mat = self._img_transform(
                    img,
                    resize=output_resize,
                    resize_dims=output_dims,
                    crop=output_crop,
                    flip=False,
                    rotate=0,
                )
                #!----------------------------------------------------#
                mat = np.eye(4)
                mat[:3, :3] = ida_mat # todo 
                new_imgs.append(np.array(img).astype(np.float32))
                # todo (wys 12.30)
                transforms.append(mat) # todo img2img' 矩阵
            
            results["output_img"] = new_imgs
            # todo (wys 12.30)
            results["output_img_aug_mat"] = np.array(transforms).astype(np.float32)

        
        return results

    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )
    # todo 对图像做2D增强，并计算“像素坐标系 -> 增强后像素坐标系” 的仿射变换举证
    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2) # todo 2x2 仿射线性部分
        ida_tran = torch.zeros(2) # todo 平移
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop) # todo 裁剪窗口左上角(crop_w,crop_h)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        resize =  torch.diag(torch.tensor(resize, dtype=ida_rot.dtype, device=ida_rot.device))
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2]) 
        if flip: 
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        
        return img, ida_mat


@TRANSFORMS.register_module()
class LoadFeatMaps(ResizeCropFlipImage):

    def __init__(self, data_root, key, apply_aug=True):
        self.data_root = data_root
        self.key = key
        self.apply_aug = apply_aug

    def __call__(self, results):
        #! 加载输入图像的深度图 跟着输入图像的预处理对深度图进行处理！
        aug_configs = results.get("aug_configs")
        resize, resize_dims, crop, flip, rotate, output_resize, output_dims, output_crop = aug_configs

        feats = [] 
        for i, filename in enumerate(results['filename']):
            feat = np.load(
                os.path.join(self.data_root,
                             filename.split('/')[-1].split('.')[0] + '.npy'))
            '''
            import cv2
            cv2.imwrite('depth_0.png',feat.astype(np.uint8))
            '''
            feat = Image.fromarray(feat)
            feat, ida_mat = self._img_transform(
                feat,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )     
            feat = np.array(feat).astype(np.float32)       
            feats.append(feat)
            '''
            import cv2
            cv2.imwrite('depth_1.png',feat.astype(np.uint8))
            '''
        
        results[self.key] = np.array(feats,dtype=np.float32)
        
        # todo 非关键帧无深度图
        # if 'output_filename' in results:
        #     feats = [] 
        #     for i, filename in enumerate(results['output_filename']): #! 'output_filename'
        #         feat = np.load(
        #             os.path.join(self.data_root,
        #                         filename.split('/')[-1].split('.')[0] + '.npy'))
        #         '''
        #         import cv2
        #         cv2.imwrite('depth_0.png',feat.astype(np.uint8))
        #         '''
        #         feat = Image.fromarray(feat)
        #         feat, ida_mat = self._img_transform(
        #             feat,
        #             resize=output_resize,
        #             resize_dims=output_dims,
        #             crop=output_crop,
        #             flip=False,
        #             rotate=0,
        #         )     
        #         feat = np.array(feat).astype(np.float32)       
        #         feats.append(feat)
        #         '''
        #         import cv2
        #         cv2.imwrite('depth_1.png',feat.astype(np.uint8))
        #         '''
            
        #     results[f'output_{self.key}'] = np.array(feats,dtype=np.float32)        
        
        
        return results
        
        
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """        
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) # todo to_rgb == True: bgr -> rgb
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        
        if "output_img" in results:
            if self.to_rgb:
                results["output_img"] = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img) for img in results["output_img"]]
            else:
                results["output_img"] = [img for img in results["output_img"]]

            '''
            import cv2
            import numpy as np
            img_norm = results["img"][0]
            mean = self.mean
            std = self.std
            input_img = (img_norm * std) + mean
            input_img = np.clip(input_img, 0, 255).astype(np.uint8)
            cv2.imwrite('input.png',input_img[...,::-1])  # RGB -> BGR
            cv2.imwrite('output.png',results["output_img"][0][...,::-1].astype(np.uint8))  # RGB -> BGR
            '''
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@TRANSFORMS.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(
                    -self.brightness_delta, self.brightness_delta
                )
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str


@TRANSFORMS.register_module()
class BEVLoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged', crop_size=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.crop_size = crop_size

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        # imgs = [mmcv.imread(name, self.color_type) for name in filename]
        
        img_bytes = [
            get(name, backend_args=None) for name in filename
        ] # todo backend_args: None  get(): 获取文件原始二进制内容
        imgs = [
            mmcv.imfrombytes(
                img_byte,
                flag=self.color_type, # todo unchanged: 保持bgr
                backend='pillow',
                channel_order='rgb') for img_byte in img_bytes
        ]   
            
        '''
        import cv2
        cv2.imwrite('output.png',imgs[0][...,::-1].astype(np.uint8)) 
        '''
        img = np.stack(imgs, axis=-1)
        
        if self.crop_size is not None:
            img = img[:self.crop_size[0], :self.crop_size[1]]
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename # todo img_path
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['ori_img'] = deepcopy(img)
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        
        # todo ---------------------------------#
        # todo 获取输出图像(用于渲染监督)
        if 'output_img_filename' in results:
            filename = results['output_img_filename']
            img_bytes = [
                get(name, backend_args=None) for name in filename
            ] # todo backend_args: None  get(): 获取文件原始二进制内容
            imgs = [
                mmcv.imfrombytes(
                    img_byte,
                    flag=self.color_type, # todo unchanged: 保持bgr
                    backend='pillow',
                    channel_order='rgb') for img_byte in img_bytes
            ]
            results['output_filename'] = filename
            results['output_img'] = imgs      
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@TRANSFORMS.register_module()
class LoadPointFromFile(object): # todo 

    def __init__(self, pc_range, num_pts, use_ego=False):
        self.use_ego = use_ego
        self.pc_range = pc_range
        self.num_pts = num_pts

    def __call__(self, results):
        pts_path = results['pts_filename']
        scan = np.fromfile(pts_path, dtype=np.float32)
        scan = scan.reshape((-1, 5))[:, :4]
        scan[:, 3] = 1.0 # n, 4
        if self.use_ego:
            ego2lidar = results['ego2lidar']
            lidar2ego = np.linalg.inv(ego2lidar)
            scan = lidar2ego[None, ...] @ scan[..., None]
            scan = np.squeeze(scan, axis=-1)
        scan = scan[:, :3] # n, 3

        ### filter
        norm = np.linalg.norm(scan, 2, axis=-1)
        mask = (scan[:, 0] > self.pc_range[0]) & (scan[:, 0] < self.pc_range[3]) & \
            (scan[:, 1] > self.pc_range[1]) & (scan[:, 1] < self.pc_range[4]) & \
            (scan[:, 2] > self.pc_range[2]) & (scan[:, 2] < self.pc_range[5]) & \
            (norm > 1.0)
        scan = scan[mask]

        ### append
        if scan.shape[0] < self.num_pts:
            multi = int(math.ceil(self.num_pts * 1.0 / scan.shape[0])) - 1
            scan_ = np.repeat(scan, multi, 0)
            scan_ = scan_ + np.random.randn(*scan_.shape) * 0.2
            scan_ = scan_[np.random.choice(scan_.shape[0], self.num_pts - scan.shape[0], False)]
            scan_[:, 0] = np.clip(scan_[:, 0], self.pc_range[0], self.pc_range[3])
            scan_[:, 1] = np.clip(scan_[:, 1], self.pc_range[1], self.pc_range[4])
            scan_[:, 2] = np.clip(scan_[:, 2], self.pc_range[2], self.pc_range[5])
            scan = np.concatenate([scan, scan_], 0)
        else:
            scan = scan[np.random.choice(scan.shape[0], self.num_pts, False)]

        scan[:, 0] = (scan[:, 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        scan[:, 1] = (scan[:, 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        scan[:, 2] = (scan[:, 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        results['anchor_points'] = scan.astype(np.float32)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str





@TRANSFORMS.register_module()
class LoadPseudoPointFromFile(object): # todo 

    def __init__(self, datapath, pc_range, num_pts, is_ego=True, use_ego=False):
        self.datapath = datapath
        self.is_ego = is_ego
        self.use_ego = use_ego
        self.pc_range = pc_range
        self.num_pts = num_pts
        pass

    def __call__(self, results):
        pts_path = os.path.join(self.datapath, f"{results['sample_idx']}.npy")
        scan = np.load(pts_path)
        if self.is_ego and (not self.use_ego):
            ego2lidar = results['ego2lidar']
            scan = np.concatenate([scan, np.ones_like(scan[:, :1])], axis=-1)
            scan = ego2lidar[None, ...] @ scan[..., None] # p, 4, 1
            scan = np.squeeze(scan, axis=-1)

        if (not self.is_ego) and self.use_ego:
            ego2lidar = results['ego2lidar']
            lidar2ego = np.linalg.inv(ego2lidar)
            scan = np.concatenate([scan, np.ones_like(scan[:, :1])], axis=-1)
            scan = lidar2ego[None, ...] @ scan[..., None]
            scan = np.squeeze(scan, axis=-1)

        scan = scan[:, :3] # n, 3

        ### filter
        norm = np.linalg.norm(scan, 2, axis=-1)
        mask = (scan[:, 0] > self.pc_range[0]) & (scan[:, 0] < self.pc_range[3]) & \
            (scan[:, 1] > self.pc_range[1]) & (scan[:, 1] < self.pc_range[4]) & \
            (scan[:, 2] > self.pc_range[2]) & (scan[:, 2] < self.pc_range[5]) & \
            (norm > 1.0)
        scan = scan[mask]

        ### append
        if scan.shape[0] < self.num_pts:
            multi = int(math.ceil(self.num_pts * 1.0 / scan.shape[0])) - 1
            scan_ = np.repeat(scan, multi, 0)
            scan_ = scan_ + np.random.randn(*scan_.shape) * 0.3
            scan_ = scan_[np.random.choice(scan_.shape[0], self.num_pts - scan.shape[0], False)]
            scan_[:, 0] = np.clip(scan_[:, 0], self.pc_range[0], self.pc_range[3])
            scan_[:, 1] = np.clip(scan_[:, 1], self.pc_range[1], self.pc_range[4])
            scan_[:, 2] = np.clip(scan_[:, 2], self.pc_range[2], self.pc_range[5])
            scan = np.concatenate([scan, scan_], 0)
        else:
            scan = scan[np.random.choice(scan.shape[0], self.num_pts, False)]

        scan[:, 0] = (scan[:, 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        scan[:, 1] = (scan[:, 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        scan[:, 2] = (scan[:, 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        results['anchor_points'] = scan

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class LoadOccupancySurroundOcc(object):

    def __init__(self, occ_path, semantic=False, use_ego=False, use_sweeps=False, perturb=False):
        self.occ_path = occ_path
        self.semantic = semantic
        self.use_ego = use_ego
        assert semantic and (not use_ego)
        self.use_sweeps = use_sweeps
        self.perturb = perturb

        xyz = self.get_meshgrid([-50, -50, -5.0, 50, 50, 3.0], [200, 200, 16], 0.5)
        self.xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1) # x, y, z, 4

    def get_meshgrid(self, ranges, grid, reso):
        xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([
            xxx, yyy, zzz
        ], dim=-1).numpy()
        return xyz # x, y, z, 3

    def __call__(self, results):
        label_file = os.path.join(self.occ_path, results['pts_filename'].split('/')[-1]+'.npy')
        if os.path.exists(label_file): # todo True
            label = np.load(label_file)

            new_label = np.ones((200, 200, 16), dtype=np.int64) * 17
            new_label[label[:, 0], label[:, 1], label[:, 2]] = label[:, 3]

            mask = new_label != 0

            results['occ_label'] = new_label if self.semantic else new_label != 17
            results['occ_cam_mask'] = mask
        elif self.use_sweeps:
            new_label = np.ones((200, 200, 16), dtype=np.int64) * 17
            mask = new_label != 0
            results['occ_label'] = new_label if self.semantic else new_label != 17
            results['occ_cam_mask'] = mask
        else:
            raise NotImplementedError

        xyz = self.xyz.copy()
        if getattr(self, "perturb", False):
            # xyz[..., :3] = xyz[..., :3] + (np.random.rand(*xyz.shape[:-1], 3) - 0.5) * (0.5 - 1e-3)
            norm_distribution = np.clip(np.random.randn(*xyz.shape[:-1], 3) / 6, -0.5, 0.5)
            xyz[..., :3] = xyz[..., :3] + norm_distribution * 0.49

        if not self.use_ego:
            occ_xyz = xyz[..., :3]
        else:
            ego2lidar = np.linalg.inv(results['ego2lidar']) # 4, 4
            occ_xyz = ego2lidar[None, None, None, ...] @ xyz[..., None] # x, y, z, 4, 1
            occ_xyz = np.squeeze(occ_xyz, -1)[..., :3]
        results['occ_xyz'] = occ_xyz
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class LoadOccupancyKITTI360(object):

    def __init__(self, occ_path, semantic=False, unknown_to_empty=False, training=False):
        self.occ_path = occ_path
        self.semantic = semantic

        xyz = self.get_meshgrid([0.0, -25.6, -2.0, 51.2, 25.6, 4.4], [256, 256, 32], 0.2)
        self.xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1) # x, y, z, 4
        self.unknown_to_empty = unknown_to_empty
        self.training = training

    def get_meshgrid(self, ranges, grid, reso):
        xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([
            xxx, yyy, zzz
        ], dim=-1).numpy()
        return xyz # x, y, z, 3

    def __call__(self, results):
        occ_xyz = self.xyz[..., :3].copy()
        results['occ_xyz'] = occ_xyz

        ## read occupancy label
        label_path = os.path.join(
            self.occ_path, results['sequence'], "{}_1_1.npy".format(results['token']))
        label = np.load(label_path).astype(np.int64)
        if getattr(self, "unknown_to_empty", False) and getattr(self, "training", False):
            label[label == 255] = 0

        results['occ_cam_mask'] = (label != 255)
        results['occ_label'] = label
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str



