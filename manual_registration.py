# examples/Python/Advanced/interactive_visualization.py

import numpy as np
import copy
import open3d as o3d
from os.path import join
import cv2
import pickle


voxel_size = 0.02
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5

depth_threshold = 800

height, width = 480, 640
render_hand_color = False

dir_images = '/data/mario2CVPR/0030'
filename_model = '/data/scriptsForAnnotation/models/037_scissors/textured_simple.ply'
start_fn = 400
end_fn = 2000
step = 50
obj_id = 17
fn = 301
#
# dir_images = '/media/mahdi/data2/hand_pose/calibration/captured_data/seq_03'
# filename_model = '/media/mahdi/data2/hand_pose/calibration/captured_data/seq_03/calibration/model/giant_duck.ply'
# obj_id = 255
# fn = 222
#
# dir_images = '/media/mahdi/data2/captured_data/temp/Shreyas_Banana_1'
# filename_model = '/media/mahdi/data2/datasets/YCB/models_simplified/011_banana/simple.ply'
# obj_id = 10
# fn = 144


# T_obj = np.loadtxt(join(dir_images, 'calibration', 'init_obj_{}.txt'.format(fn)))

# filename_mano_vertex_color = '/media/mahdi/data2/GenHOPE/models/vertex_colors.npy'



# filename_model = '/media/mahdi/data2/datasets/YCB/models_simplified/021_bleach_cleanser/simple.ply'


def inverse_relative(pose_1_to_2):
    pose_2_to_1 = np.zeros((4, 4), dtype='float32')
    pose_2_to_1[:3, :3] = np.transpose(pose_1_to_2[:3, :3])
    pose_2_to_1[:3, 3:4] = -np.dot(np.transpose(pose_1_to_2[:3, :3]), pose_1_to_2[:3, 3:4])
    pose_2_to_1[3, 3] = 1
    return pose_2_to_1


def get_intrinsics(filename):
    K = np.zeros((3, 3), dtype='float32')
    f = open(filename, "r")
    intrinsics = f.read()
    print(intrinsics)
    f.close()

    keywords = ['fx', 'ppx', 'fy', 'ppy']
    indices = [(0, 0), (0, 2), (1, 1), (1, 2)]
    for idx, keyword in enumerate(keywords):
        sub_string = intrinsics[intrinsics.find(keyword) + len(keyword) + 2:]
        sub_string = sub_string[:sub_string.find(',')]

        K[indices[idx]] = float(sub_string)

    K[2, 2] = 1
    np.savetxt(filename.replace('intrinsics', 'intrinsics_3x3'), K)
    return K


dir_calibration = join(dir_images, 'calibration')
depth_scale = 1./np.loadtxt(join(dir_calibration, 'cam_0_depth_scale.txt'))


cams_order = np.loadtxt(join(dir_images, 'cam_orders.txt')).astype('uint8').tolist()
# cams_order = np.loadtxt(join(dir_images, 'cam_orders.txt')).astype('uint8').tolist()
Ts = []
for i in range(len(cams_order)):
    T_i = np.loadtxt(join(dir_calibration, 'trans_{}.txt'.format(i)))
    Ts.append(T_i)


T_gl_cv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).astype('float32')
coordChangMat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype('float32')



def load_point_clouds(voxel_size=0.0, frame_number=1):
    pcds = []

    # mano_color = np.load(filename_mano_vertex_color)[:, [2, 1, 0]]

    for i in range(len(cams_order)):

        # if i == 2:
        #     continue
        path_color = join(dir_images, 'rgb', '{}'.format(cams_order[i]), '{:05d}.png'.format(frame_number))

        color_raw = o3d.io.read_image(path_color)
        depth_raw = o3d.io.read_image(path_color.replace('rgb', 'depth'))
        depth_temp = np.array(depth_raw)
        indices = np.where(depth_temp > depth_threshold)
        depth_temp[indices] = 0

        '''
        path_mask = join(dir_images, 'vis', '{}'.format(cams_order[i]), 'raw_seg_results', '{:05d}.png'.format(frame_number))
        print(path_mask)
        # path_mask = join(dir_images, 'masks', '{}.png'.format(cams_order[i]))
        # print path_mask
        mask = cv2.imread(path_mask)
        # print(mask[:, :, 0])
        indices = np.where(mask[:, :, 0] != obj_id)
        # mask[indices] = 255
        # cv2.imshow('mask', mask)
        # cv2.waitKey()

        depth_temp[indices] = 0
        
        '''
        depth_raw = o3d.Image(depth_temp)


        # K = np.loadtxt(join(dir_calibration, 'cam_{}_intrinsics_3x3.txt'.format(i))).tolist()
        K = get_intrinsics(join(dir_calibration, 'cam_{}_intrinsics.txt'.format(i))).tolist()
        rgbd_image = o3d.geometry.create_rgbd_image_from_color_and_depth(color_raw,
                                                                         depth_raw,
                                                                         depth_scale=depth_scale,
                                                                         convert_rgb_to_intensity=False)

        pcd = o3d.geometry.create_point_cloud_from_rgbd_image(
            rgbd_image, o3d.open3d.camera.PinholeCameraIntrinsic(width=width,
                                                                 height=height,
                                                                 fx=K[0][0],
                                                                 fy=K[1][1],
                                                                 cx=K[0][2],
                                                                 cy=K[1][2]))

        pcd.transform((Ts[i]))


        o3d.geometry.estimate_normals(pcd,
                                      search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
                                                                                        max_nn=30))
        # pcd.transform(T_gl_cv)

        pcds.append(pcd)


    # xyz = np.loadtxt('/data/mario2CVPR/0012/manual_registration/00620.txt')
    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(xyz)
    #
    #
    #
    # # pcd.transform(T_gl_cv)
    # pcd1.transform(np.linalg.inv(Ts[0]))
    # pcd1.transform(T_gl_cv)
    #
    # #pcds = [pcd]
    # # pcds.append(pcd)
    #
    #
    #
    # with open('/data/mario2CVPR/0012/manual_registration/test/00620.pkl', 'rb') as f:
    #     est = pickle.load(f)
    #
    # xyz = est['JTransformed'][[17,18,19,20,16]]#.dot(coordChangMat)
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(xyz)
    #
    # print(np.linalg.norm(np.array(pcd2.points) - np.array(pcd1.points), axis=1))
    #
    # draw_registration_result_2(pcd1, pcd2)


    return pcds


def draw_registration_result_2(source, target):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    o3d.visualization.draw_geometries([source_temp, target_temp])

def draw_registration_result(source):
    source_temp = copy.deepcopy(source)
    o3d.visualization.draw_geometries([source_temp])


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def combine_point_clouds(pcds):
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcd_combined += pcds[point_id]

    return pcd_combined


def manual_registration():
    for frame_number in range(start_fn, end_fn, step):
        pcds = load_point_clouds(frame_number=frame_number)
        pcd = combine_point_clouds(pcds)

        print("Visualization of two point clouds before manual alignment")
        # draw_registration_result(pcd)

        # pick points from point cloud
        picked_id_source = pick_points(pcd)
        finger_tips = np.array(pcd.points)[picked_id_source, :]
        print(picked_id_source)
        np.savetxt(join(dir_images, 'manual_registration', '{:05d}.txt'.format(frame_number)), finger_tips)

if __name__ == "__main__":
    manual_registration()