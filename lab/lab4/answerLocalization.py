from typing import List
import numpy as np
from utils import Particle
from copy import deepcopy
### 可以在这里写下一些你需要的变量和函数 ###
COLLISION_DISTANCE = 1
MAX_ERROR = 50000
K = 0.4
NOISE_STD = 0.1
NOISE_STD_THETA = 0.1

def dist_to_edge(x, y, walls):
    ret = 999999
    for wall in walls:
        wall_x, wall_y = wall  # 墙壁中心坐标
        # 墙壁边界
        left, right = wall_x - 0.5, wall_x + 0.5
        bottom, top = wall_y - 0.5, wall_y + 0.5
        # 计算点到墙壁边缘的最小距离
        dist_x = max(left - x, 0, x - right)
        dist_y = max(bottom - y, 0, y - top)
        dist_to_edge = np.sqrt(dist_x**2 + dist_y**2)
        ret = min(ret, dist_to_edge)
    return ret 

def is_valid_point(x, y, walls):
    x_min, x_max = np.min(walls[:, 0]), np.max(walls[:, 0])
    y_min, y_max = np.min(walls[:, 1]), np.max(walls[:, 1])
    return (x <= x_max) and (x >= x_min) and (y <= y_max) and (y >= y_min)

def add_noise(particle, walls):
    i=0
    while(True):
        #print(i)
        i+=1
        new_position = particle.position + np.random.normal(0, NOISE_STD, size=particle.position.shape)
        new_orientation = particle.theta + np.random.normal(0, NOISE_STD_THETA)
        #print(new_position, dist_to_edge(new_position[0], new_position[1], walls))
        # if (is_valid_point(new_position[0], new_position[1], walls) and (new_orientation >= -np.pi) and (new_orientation <= np.pi)):
        #     
        return Particle(new_position[0], new_position[1], new_orientation, particle.weight)
### 可以在这里写下一些你需要的变量和函数 ###


def generate_uniform_particles(walls, N):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    N: int, 采样点数量
    输出：
    particles: List[Particle], 返回在空地上均匀采样出的N个采样点的列表，每个点的权重都是1/N
    """
    all_particles: List[Particle] = []
        # all_particles.append(Particle(1.0, 1.0, 1.0, 0.0))
    ### 你的代码 ###
    x_min, x_max = np.min(walls[:, 0]), np.max(walls[:, 0])
    y_min, y_max = np.min(walls[:, 1]), np.max(walls[:, 1])

    count = 0
    while count < N:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        if is_valid_point(x, y, walls) and dist_to_edge(x, y, walls) >= 1e-6:
            theta = np.random.uniform(-np.pi, np.pi)
            all_particles.append(Particle(x, y, theta, 1.0/N))
            count += 1

    ### 你的代码 ###
    return all_particles


def calculate_particle_weight(estimated, gt):
    """
    输入：
    estimated: np.array, 该采样点的距离传感器数据
    gt: np.array, Pacman实际位置的距离传感器数据
    输出：
    weight, float, 该采样点的权重
    """
    weight = 1.0
    ### 你的代码 ###
    weight = np.exp(-K * np.linalg.norm((estimated - gt)))
    ### 你的代码 ###
    return weight


def resample_particles(walls, particles: List[Particle]):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    particles: List[Particle], 上一次采样得到的粒子，注意是按权重从大到小排列的
    输出：
    particles: List[Particle], 返回重采样后的N个采样点的列表
    """
    N = len(particles)
    weights = np.array([p.weight for p in particles])
    sample_num = np.floor(weights * N).astype(int)
    # weights /= np.sum(weights)
    
    # indices = np.random.choice(range(N), size=N, replace=True, p=weights)########
    # resampled_particles = []
    # for i in indices:
    #     valid = False
    #     while not valid:
    #         # 在原有粒子上加的高斯噪声
    #         noise = np.random.normal(0, NOISE_STD, size=particles[i].position.shape)
    #         new_position = particles[i].position + noise
    #         if is_valid_point(new_position[0], new_position[1], walls):
    #             new_orientation = particles[i].theta + np.random.normal(0, NOISE_STD)
    #             resampled_particles.append(Particle(new_position[0], new_position[1], new_orientation))
    #             valid = True
    resampled_particles: List[Particle] = []
        # resampled_particles.append(Particle(1.0, 1.0, 1.0, 0.0))
    ### 你的代码 ###
    
    for i, particle in enumerate(particles):
        # print(i)
        # print(sample_num[i])
        for _ in range(sample_num[i]):
            resampled_particles.append(add_noise(particle, walls))
    rest_num = N - len(resampled_particles)
    rest_particles = generate_uniform_particles(walls, rest_num)
    resampled_particles += rest_particles
    # ### 你的代码 ###
    return resampled_particles

def apply_state_transition(p: Particle, traveled_distance, dtheta):
    """
    输入：
    p: 采样的粒子
    traveled_distance, dtheta: ground truth的Pacman这一步相对于上一步运动方向改变了dtheta，并移动了traveled_distance的距离
    particle: 按照相同方式进行移动后的粒子
    """
    ### 你的代码 ###
    p.theta += dtheta
    dx = traveled_distance * np.cos(p.theta)
    dy = traveled_distance * np.sin(p.theta)
    p.position += np.array([dx, dy])
    ### 你的代码 ###
    return p

def get_estimate_result(particles: List[Particle]):
    """
    输入：
    particles: List[Particle], 全部采样粒子
    输出：
    final_result: Particle, 最终的猜测结果
    """
    final_result = Particle()
    ### 你的代码 ###
    weights = np.array([p.get_weight() for p in particles])
    final_result = particles[np.argmax(weights)]
    ### 你的代码 ###
    return final_result