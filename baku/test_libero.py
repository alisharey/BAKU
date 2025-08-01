
import warnings
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
# os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark, get_libero_path
from PIL import Image
import numpy as np

benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()

# retrieve a specific task
task_id = 4
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
      f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

# step over the environment
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()

import numpy as np

def quaternion_multiply(q1, q2):
    """Multiply two quaternions [x, y, z, w]"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    
    return np.array([x, y, z, w])

# Base rotation
base_quat = np.array([0.82722725, 0.49892427, 0.16778389, 0.24274413])

# 90-degree rotations around each axis (in camera frame)
rot_90_x = np.array([0.7071068, 0, 0, 0.7071068])  # 90° around X
rot_90_y = np.array([0, 0.7071068, 0, 0.7071068])  # 90° around Y  
rot_90_z = np.array([0, 0, 0.7071068, 0.7071068])  # 90° around Z

rotations = { 'right_medium': np.array([0.82722725, 0.49892427, 0.16778389, 0.24274413]),
    # 'neg_xz_x90_base': base_quat,
    # 'neg_xz_x90_rot_x90': quaternion_multiply(rot_90_x, base_quat),
    # 'neg_xz_x90_rot_y90': quaternion_multiply(rot_90_y, base_quat),
    # 'neg_xz_x90_rot_z90': quaternion_multiply(rot_90_z, base_quat),
    # 'neg_xz_x90_rot_x180': quaternion_multiply(quaternion_multiply(rot_90_x, rot_90_x), base_quat),
    # 'neg_xz_x90_rot_y180': quaternion_multiply(quaternion_multiply(rot_90_y, rot_90_y), base_quat),
    # 'neg_xz_x90_rot_z180': quaternion_multiply(quaternion_multiply(rot_90_z, rot_90_z), base_quat),
    # 'neg_xz_x90_rot_x270': quaternion_multiply(quaternion_multiply(quaternion_multiply(rot_90_x, rot_90_x), rot_90_x), base_quat),
    # 'neg_xz_x90_rot_y270': quaternion_multiply(quaternion_multiply(quaternion_multiply(rot_90_y, rot_90_y), rot_90_y), base_quat),
    # 'neg_xz_x90_rot_z270': quaternion_multiply(quaternion_multiply(quaternion_multiply(rot_90_z, rot_90_z), rot_90_z), base_quat)
}
camera_name = 'agentview'
cam_id = env.sim.model.camera_name2id(camera_name)
old_position = env.sim.model.cam_pos[cam_id].copy()
delta_pos = np.array([-0.1, -0.7, -0.1])  # Adjusted delta position
base_position = old_position + delta_pos

for rotation_name, rotation_quat in rotations.items():
    # Reset environment
    env.reset()
    
    # Set camera position and rotation
    env.sim.model.cam_pos[cam_id] = base_position
    env.sim.model.cam_quat[cam_id] = rotation_quat
    
    # Take one step and save image
    dummy_action = [0.] * 7
    obs, reward, done, info = env.step(dummy_action)
    
    if "agentview_image" in obs:
        img_array = obs["agentview_image"]
        img_array = np.flipud(img_array)
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)
        img = Image.fromarray(img_array)
        img.save(f"rotation_{rotation_name}_{delta_pos}.png")
        print(f"Saved image for rotation: {rotation_name}")

env.close()