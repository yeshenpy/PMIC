import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time


class Water():
    def __init__(self, init_x, init_y, init_max_velocity, max_y, min_y, max_x):
        self.init_x, self.init_y = init_x, init_y
        self.x, self.y = init_x, init_y
        self.max_y, self.min_y, self.max_x = max_y, min_y, max_x
        self.velocity = init_max_velocity * (self.x / self.max_x - (self.x / self.max_x) ** 2)

    def update_velocity(self, max_velocity):
        self.velocity = max_velocity * (self.x / self.max_x - (self.x / self.max_x) ** 2)

    def move(self, delta_t=1.):
        self.y += self.velocity * delta_t
        if self.y > self.max_y:
            self.y = self.min_y


class ShipAcrossRiverHarder(gym.Env):
    def __init__(self, state_normalize=True):
        self.state_normalize = state_normalize
        # 设置状态空间，动作空间
        self.min_x, self.max_x = 0, 50
        self.min_y, self.max_y = 0, 100
        self.min_angle, self.max_angle = -np.pi / 3, np.pi / 3
        self.min_velocity, self.max_velocity = 2, 5
        self.min_angle_velocity, self.max_angle_velocity = -1, 1
        observation_space_low = np.array(
            [self.min_x, self.min_y, self.min_angle, self.min_velocity, self.min_angle_velocity])
        observation_space_high = np.array(
            [self.max_x, self.max_y, self.max_angle, self.max_velocity, self.max_angle_velocity])
        self.observation_space = spaces.Box(observation_space_low, observation_space_high)

        self.min_acc_velocity, self.max_acc_velocity = -2, 2
        self.min_angle_velocity, self.max_angle_velocity = -1, 1
        action_space_low = np.array([self.min_acc_velocity, self.min_angle_velocity])
        action_space_high = np.array([self.max_acc_velocity, self.max_angle_velocity])
        self.action_space = spaces.Box(action_space_low, action_space_high)

        # 设置小船及水流初始状态
        self.reset()
        self._init_water()

        # 绘制屏幕
        self.viewer = None

        # 设置 goal 位置
        self.optimal_y, self.optimal_range = 30, 5
        self.sub_optimal_1_y, self.sub_optimal_1_range = 80, 20
        self.sub_optimal_2_y, self.sub_optimal_2_range = 50, 10
        print(3 * (self.optimal_range - 0))
        print((self.sub_optimal_1_range - 0) * 0.5)
        print(self.sub_optimal_2_range - 0)

    def _init_water(self):
        self.waters = [Water(init_x, init_y, self.max_water_speed, self.max_y, self.min_y, self.max_x) for init_x in
                       range(2, self.max_x, 5) for init_y in range(0, self.max_y, 20)]

    def seed(self, seed=None):
        # 产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _set_max_water_speed(self):
        '''
        水流速度是二次函数，中间高，两边低
        :return:
        '''
        self.max_water_speed = 4 + 1 * np.random.standard_normal()

    def reset(self):
        # 设置水流速度
        self._set_max_water_speed()

        # 设置小船初始状态
        self.x = 0
        self.y = 50
        self.angle = 0
        self.velocity = 2
        self.angle_velocity = 0

        return self._get_state()

    def _get_state(self):
        if self.state_normalize:
            return np.array([(self.x - self.min_x) / (self.max_x - self.min_x),
                             (self.y - self.min_y) / (self.max_y - self.min_y),
                             (self.angle - self.min_angle) / (self.max_angle - self.min_angle),
                             (self.velocity - self.min_velocity) / (self.max_velocity - self.min_velocity),
                             (self.angle_velocity - self.min_angle_velocity) / (
                                     self.max_angle_velocity - self.min_angle_velocity)])
        else:
            return [self.x, self.y, self.angle, self.velocity, self.angle_velocity]

    def step(self, action):
        # 求和代替积分，描绘小船变化
        time_interval = 1.
        n_split = 100
        delta_t = time_interval / n_split
        acc_velocity = action[0]
        acc_angle_velocity = action[1]
        old_x = self.x
        for n in range(n_split):
            # 小船线速度变化
            new_velocity = np.clip(self.velocity + delta_t * acc_velocity, self.min_velocity, self.max_velocity)
            # 小船角速度变化
            new_angle_velocity = np.clip(self.angle_velocity + delta_t * acc_angle_velocity, self.min_angle_velocity,
                                         self.max_angle_velocity)
            # 小船角度变化
            new_angle = np.clip(self.angle + self.angle_velocity * delta_t + 1 / 2 * acc_angle_velocity * delta_t ** 2,
                                self.min_angle, self.max_angle)
            # 当前位置水流速度
            self.water_speed = self.max_water_speed * (self.x / self.max_x - (self.x / self.max_x) ** 2)
            for water in self.waters:
                water.update_velocity(self.max_water_speed)
                water.move(delta_t=delta_t)
            # 小船 x 坐标变化
            cos_theta = np.cos((self.angle + new_angle) / 2)
            vx = self.velocity * cos_theta
            new_vx = new_velocity * cos_theta
            self.x = self.x + (vx + new_vx) / 2 * delta_t
            # 小船 y 坐标变化
            sin_theta = np.sin((self.angle + new_angle) / 2)
            vy = self.velocity * sin_theta
            new_vy = new_velocity * sin_theta
            self.y = self.y + (vy + new_vy) / 2 * delta_t + self.water_speed * delta_t
            # 更新小船速度，角度，角速度
            self.velocity = new_velocity
            self.angle = new_angle
            self.angle_velocity = new_angle_velocity

        new_x = self.x
        # print(new_x - old_x)
        # 计算reward
        reward, done = self._cal_reward()
        info = {}

        # 每一步都改变水流状态
        self._set_max_water_speed()

        return self._get_state(), reward, done, info

    def _cal_reward(self):
        # step cost
        if self.x > self.min_x and self.x < self.max_x and self.y > self.min_y and self.y < self.max_y:
            return -0.1, False
        # global optimal is at position 30
        elif self.x >= self.max_x and abs(self.y - self.optimal_y) <= self.optimal_range:
            return 3 * (self.optimal_range - abs(self.y - self.optimal_y)), True
        # sub optimal is at position 50
        elif self.x >= self.max_x and abs(self.y - self.sub_optimal_2_y) <= self.sub_optimal_2_range:
            return self.sub_optimal_2_range - abs(self.y - self.sub_optimal_2_y), True
        # sub optimal is at position 80
        elif self.x >= self.max_x and abs(self.y - self.sub_optimal_1_y) <= self.sub_optimal_1_range:
            return (self.sub_optimal_1_range - abs(self.y - self.sub_optimal_1_y)) * 0.5, True
        # out of bounds
        elif self.y <= self.min_y or self.y >= self.max_y or self.x <= self.min_x:
            return -1, True
        else:  # 正常通过右侧
            assert self.y < self.max_y and self.y > self.min_y and self.x >= self.max_x
            return 0, True

    def _convert_xy2location(self, x, y, scale, margin):
        location_y = (x - self.min_x) * scale + margin
        location_x = (self.max_y - y) * scale + margin
        return [location_x, location_y]

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400
        margin = 10
        scale = screen_height / (self.max_x - self.min_x)
        ship_width = 20
        ship_height = 40

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width + margin * 2, screen_height + margin * 2)

            # 绘制河流
            self.lower_boundary = rendering.make_polyline(
                [self._convert_xy2location(self.min_x, self.min_y, scale=scale, margin=margin),
                 self._convert_xy2location(self.min_x, self.max_y, scale=scale, margin=margin)])
            self.lower_boundary.set_linewidth(4)
            self.lower_boundary.set_color(0xD3 / 255, 0xD3 / 255, 0xD3 / 255)
            self.upper_boundary = rendering.make_polyline(
                [self._convert_xy2location(self.max_x, self.min_y, scale=scale, margin=margin),
                 self._convert_xy2location(self.max_x, self.max_y, scale=scale, margin=margin)])
            self.upper_boundary.set_linewidth(8)
            self.upper_boundary.set_color(0xD3 / 255, 0xD3 / 255, 0xD3 / 255)
            self.viewer.add_geom(self.lower_boundary)
            self.viewer.add_geom(self.upper_boundary)

            # 绘制河水
            l, r, t, b = -20, 10, 2, 0
            self.water_trans = []
            for water_data in self.waters:
                water = rendering.FilledPolygon([(l, b), (l + 2, t), (r - 2, t), (r, b)])
                water.set_color(0.63529412, 0.81176471, 0.99607843)
                tran = rendering.Transform()
                self.water_trans.append(tran)
                water.add_attr(tran)
                self.viewer.add_geom(water)

            # 绘制初始码头
            quay_left = rendering.make_circle(8)
            quay_left.set_color(1, 0, 0, 0.5)
            quay_transform = rendering.Transform()
            quay_left.add_attr(quay_transform)
            quay_transform.set_translation(*self._convert_xy2location(self.min_x, self.max_y/2, scale=scale, margin=margin))
            self.viewer.add_geom(quay_left)


            # 绘制三个goal
            # optimal goal
            self.optimal_goal = rendering.make_polyline(
                [self._convert_xy2location(self.max_x, self.optimal_y - self.optimal_range, scale=scale, margin=margin),
                 self._convert_xy2location(self.max_x, self.optimal_y + self.optimal_range, scale=scale,
                                           margin=margin)])
            self.optimal_goal.set_linewidth(8)
            self.optimal_goal.set_color(*[0.35, 0.85, 0.35, 0.8])
            self.viewer.add_geom(self.optimal_goal)
            # reward function
            global_reward = rendering.make_polyline(
                [self._convert_xy2location(self.max_x, self.optimal_y - self.optimal_range, scale=scale, margin=margin),
                 self._convert_xy2location(self.max_x, self.optimal_y + self.optimal_range, scale=scale, margin=margin),
                 self._convert_xy2location(self.max_x - 15, self.optimal_y, scale=scale, margin=margin),
                 self._convert_xy2location(self.max_x, self.optimal_y - self.optimal_range, scale=scale, margin=margin)
                 ])
            global_reward.set_color(*[0.35, 0.85, 0.35, 0.8])
            self.viewer.add_geom(global_reward)
            global_reward.set_linewidth(8)
            self.viewer.add_geom(global_reward)
            # center point
            optimal_goal_center = rendering.make_circle(8)
            optimal_goal_center.set_color(1, 0, 0, 0.5)
            optimal_goal_center_transform = rendering.Transform()
            optimal_goal_center.add_attr(optimal_goal_center_transform)
            optimal_goal_center_transform.set_translation(
                *self._convert_xy2location(self.max_x, self.optimal_y, scale=scale, margin=margin))
            self.viewer.add_geom(optimal_goal_center)



            # sub optimal goal 1
            self.sub_optimal_goal_1 = rendering.make_polyline(
                [self._convert_xy2location(self.max_x, self.sub_optimal_1_y - self.sub_optimal_1_range, scale=scale,
                                           margin=margin),
                 self._convert_xy2location(self.max_x, self.sub_optimal_1_y + self.sub_optimal_1_range, scale=scale,
                                           margin=margin)])
            self.sub_optimal_goal_1.set_linewidth(8)
            # self.sub_optimal_goal.set_color(*[0.85, 0.35, 0.35])
            self.sub_optimal_goal_1.set_color(*[0xfd / 255, 0xdc / 255, 0x5c / 255, 0.8])
            self.viewer.add_geom(self.sub_optimal_goal_1)
            # reward function
            sub_reward1 = rendering.make_polyline(
                [self._convert_xy2location(self.max_x, self.sub_optimal_1_y - self.sub_optimal_1_range, scale=scale,
                                           margin=margin),
                 self._convert_xy2location(self.max_x, self.sub_optimal_1_y + self.sub_optimal_1_range, scale=scale,
                                           margin=margin),
                 self._convert_xy2location(self.max_x - 10, self.sub_optimal_1_y, scale=scale,
                                           margin=margin),
                 self._convert_xy2location(self.max_x, self.sub_optimal_1_y - self.sub_optimal_1_range, scale=scale,
                                           margin=margin)
                 ])
            sub_reward1.set_color(*[0xfd / 255, 0xdc / 255, 0x5c / 255])
            self.viewer.add_geom(sub_reward1)
            sub_reward1.set_linewidth(8)
            self.viewer.add_geom(sub_reward1)
            # center point
            sub_optimal_goal_center = rendering.make_circle(8)
            sub_optimal_goal_center.set_color(1, 0, 0, 0.5)
            sub_optimal_goal_center_transform = rendering.Transform()
            sub_optimal_goal_center.add_attr(sub_optimal_goal_center_transform)
            sub_optimal_goal_center_transform.set_translation(
                *self._convert_xy2location(self.max_x, self.sub_optimal_1_y, scale=scale, margin=margin))
            self.viewer.add_geom(sub_optimal_goal_center)

            # sub optimal goal 2
            self.sub_optimal_goal_2 = rendering.make_polyline(
                [self._convert_xy2location(self.max_x, self.sub_optimal_2_y - self.sub_optimal_2_range, scale=scale,
                                           margin=margin),
                 self._convert_xy2location(self.max_x, self.sub_optimal_2_y + self.sub_optimal_2_range, scale=scale,
                                           margin=margin)])
            self.sub_optimal_goal_2.set_linewidth(8)
            # self.sub_optimal_goal.set_color(*[0.85, 0.35, 0.35])
            self.sub_optimal_goal_2.set_color(*[0xfd / 255, 0xdc / 255, 0x5c / 255, 0.8])
            self.viewer.add_geom(self.sub_optimal_goal_2)
            # reward function
            sub_reward2 = rendering.make_polyline(
                [self._convert_xy2location(self.max_x, self.sub_optimal_2_y - self.sub_optimal_2_range, scale=scale,
                                           margin=margin),
                 self._convert_xy2location(self.max_x, self.sub_optimal_2_y + self.sub_optimal_2_range, scale=scale,
                                           margin=margin),
                 self._convert_xy2location(self.max_x - 10, self.sub_optimal_2_y, scale=scale,
                                           margin=margin),
                 self._convert_xy2location(self.max_x, self.sub_optimal_2_y - self.sub_optimal_2_range, scale=scale,
                                           margin=margin)
                 ])
            sub_reward2.set_color(*[0xfd / 255, 0xdc / 255, 0x5c / 255])
            self.viewer.add_geom(sub_reward2)
            sub_reward2.set_linewidth(8)
            # center point
            sub_optimal_goal_center2 = rendering.make_circle(8)
            sub_optimal_goal_center2.set_color(1, 0, 0, 0.5)
            sub_optimal_goal_center_transform2 = rendering.Transform()
            sub_optimal_goal_center2.add_attr(sub_optimal_goal_center_transform2)
            sub_optimal_goal_center_transform2.set_translation(
                *self._convert_xy2location(self.max_x, self.sub_optimal_2_y, scale=scale, margin=margin))
            self.viewer.add_geom(sub_optimal_goal_center2)

            # 绘制船
            l, r, t, b = -ship_width / 2, ship_width / 2, ship_height, 0
            ship = rendering.FilledPolygon([(l, b), (l + 2, t), (0, t + r / 1.6), (r - 2, t), (r, b)])
            ship.set_color(*[0x49 / 255, 0x75 / 255, 0x9c / 255])
            self.ship_trans = rendering.Transform()
            ship.add_attr(self.ship_trans)
            self.viewer.add_geom(ship)

        self.ship_trans.set_translation(*self._convert_xy2location(self.x, self.y, scale=scale, margin=margin))
        self.ship_trans.set_rotation(self.angle)
        for water, water_tran in zip(self.waters, self.water_trans):
            water_tran.set_translation(*self._convert_xy2location(water.x, water.y, scale=scale, margin=margin))
        time.sleep(0.2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        # print("RGB array")

    def close(self):
        pass


def test_env():
    env = ShipAcrossRiverHarder()
    agent = lambda ob: env.action_space.sample()
    ob = env.reset()
    for _ in range(1000):
        a = agent(ob)
        env.render()
        assert env.action_space.contains(a)
        (ob, reward, done, _info) = env.step(a)
        print(ob)

        if done:
            ob = env.reset()
    env.close()


if __name__ == "__main__":
    test_env()
