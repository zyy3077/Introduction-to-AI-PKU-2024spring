import numpy as np
from typing import List
from utils import TreeNode
from simuScene import PlanningMap


### 定义一些你需要的变量和函数 ###
STEP_DISTANCE = 1
TARGET_THREHOLD = 1
MAX_NODES = 10000
TARGET_REPEAT = 5
### 定义一些你需要的变量和函数 ###


class RRT:
    def __init__(self, walls) -> None:
        """
        输入包括地图信息，你需要按顺序吃掉的一列事物位置 
        注意：只有按顺序吃掉上一个食物之后才能吃下一个食物，在吃掉上一个食物之前Pacman经过之后的食物也不会被吃掉
        """
        self.map = PlanningMap(walls)
        self.walls = walls
        
        # 其他需要的变量
        ### 你的代码 ###      
        self.current_target_index = 0  
        self.calls_for_current_target = 0
        self.x_min, self.x_max = np.min(walls[:, 0]), np.max(walls[:, 0])
        self.y_min, self.y_max = np.min(walls[:, 1]), np.max(walls[:, 1])
        ### 你的代码 ###
        
        # 如有必要，此行可删除
        self.path = None
        
        
    def find_path(self, current_position, next_food):
        """
        在程序初始化时，以及每当 pacman 吃到一个食物时，主程序会调用此函数
        current_position: pacman 当前的仿真位置
        next_food: 下一个食物的位置
        
        本函数的默认实现是调用 build_tree，并记录生成的 path 信息。你可以在此函数增加其他需要的功能
        """
        
        ### 你的代码 ###  
        if self.map.checkline(list(current_position), list(np.array(next_food).astype(float)))[0] == False:
            self.path = [next_food]
        ### 你的代码 ###
        # 如有必要，此行可删除
        else:
            self.path = self.build_tree(current_position, next_food)
        
    def get_target(self, current_position, current_velocity):
        """
        主程序将在每个仿真步内调用此函数，并用返回的位置计算 PD 控制力来驱动 pacman 移动
        current_position: pacman 当前的仿真位置
        current_velocity: pacman 当前的仿真速度
        一种可能的实现策略是，仅作为参考：
        （1）记录该函数的调用次数
        （2）假设当前 path 中每个节点需要作为目标 n 次
        （3）记录目前已经执行到当前 path 的第几个节点，以及它的执行次数，如果超过 n，则将目标改为下一节点
        
        你也可以考虑根据当前位置与 path 中节点位置的距离来决定如何选择 target
        
        同时需要注意，仿真中的 pacman 并不能准确到达 path 中的节点。你可能需要考虑在什么情况下重新规划 path
        """
        target_pose = np.zeros_like(current_position)
        ### 你的代码 ###
        if self.calls_for_current_target >= TARGET_REPEAT:
            self.current_target_index += 1
            self.calls_for_current_target = 0  # 重置为下一个目标的调用次数
            # 确保索引不会超出路径长度
            if self.current_target_index >= len(self.path):
                  self.find_path(current_position, self.path[-1])
                  self.current_target_index = 0
                  self.calls_for_current_target = 0
        self.calls_for_current_target += 1
        target_pose = self.path[self.current_target_index]
        ### 你的代码 ###
        #print(target_pose)
        return target_pose
        
    ### 以下是RRT中一些可能用到的函数框架，全部可以修改，当然你也可以自己实现 ###
    def build_tree(self, start, goal):
        """
        实现你的快速探索搜索树，输入为当前目标食物的编号，规划从 start 位置食物到 goal 位置的路径
        返回一个包含坐标的列表，为这条路径上的pd targets
        你可以调用find_nearest_point和connect_a_to_b两个函数
        另外self.map的checkoccupy和checkline也可能会需要，可以参考simuScene.py中的PlanningMap类查看它们的用法
        """
        path = []
        graph: List[TreeNode] = []
        graph.append(TreeNode(-1, start[0], start[1]))
        ### 你的代码 ###
        while True:
            rand_point = np.array([np.random.uniform(self.x_min, self.x_max), np.random.uniform(self.y_min, self.y_max)])
            nearest_idx, nearest_distance = self.find_nearest_point(rand_point, graph)
            nearest_point = graph[nearest_idx]
            is_empty, new_point = self.connect_a_to_b(nearest_point.pos, rand_point)
            if is_empty:
                new_node = TreeNode(nearest_idx, new_point[0], new_point[1])
                #print(new_node.pos)
                graph.append(new_node)
                if self.map.checkline(list(new_point), list(np.array(goal).astype(float)))[0] == False or np.linalg.norm(new_point - goal) <= TARGET_THREHOLD:
                    path.append((goal[0], goal[1]))
                    current_node = new_node
                    while current_node.parent_idx != -1: 
                        path.append((current_node.pos[0], current_node.pos[1]))
                        current_node = graph[current_node.parent_idx]
                    path.append((current_node.pos[0], current_node.pos[1]))
                    path.reverse()  
                    break
        #print(path)
        ### 你的代码 ###
        return path

    @staticmethod
    def find_nearest_point(point, graph):
        """
        找到图中离目标位置最近的节点，返回该节点的编号和到目标位置距离、
        输入：
        point：维度为(2,)的np.array, 目标位置
        graph: List[TreeNode]节点列表
        输出：
        nearest_idx, nearest_distance 离目标位置距离最近节点的编号和距离
        """
        nearest_idx = -1
        nearest_distance = 10000000.
        ### 你的代码 ###
        for i in range(len(graph)):
            distance = np.linalg.norm(graph[i].pos - point)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_idx = i
        ### 你的代码 ###
        return nearest_idx, nearest_distance
    
    def connect_a_to_b(self, point_a, point_b):
        """
        以A点为起点，沿着A到B的方向前进STEP_DISTANCE的距离，并用self.map.checkline函数检查这段路程是否可以通过
        输入：
        point_a, point_b: 维度为(2,)的np.array，A点和B点位置，注意是从A向B方向前进
        输出：
        is_empty: bool，True表示从A出发前进STEP_DISTANCE这段距离上没有障碍物
        newpoint: 从A点出发向B点方向前进STEP_DISTANCE距离后的新位置，如果is_empty为真，之后的代码需要把这个新位置添加到图中
        """
        is_empty = False
        newpoint = np.zeros(2)
        ### 你的代码 ###
        newpoint = point_a + STEP_DISTANCE * (point_b - point_a) / np.linalg.norm(point_b - point_a)
        if self.map.checkline(list(point_a), list(newpoint)) == (False, None) and self.map.checkoccupy(newpoint) == False:
            is_empty = True
        ### 你的代码 ###
        return is_empty, newpoint
