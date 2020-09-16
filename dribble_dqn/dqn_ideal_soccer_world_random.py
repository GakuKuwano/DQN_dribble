import sys
sys.path.append('../scripts/')
#from kf_soccer import *
from dqn_ideal_soccer_robot import *
from tool import *
import math
import itertools
import random

class Behaivor(object):
    def operators(self, player): pass


class Goal:
    def __init__(self, x, y, radius=0.2, value=200):
        self.pos = np.array([x, y]).T
        self.radius = radius
        self.value = value
        self.right_pole = self.pos[1] + 1.3
        self.left_pole = self.pos[1] - 1.3
 
    def inside(self, pose):
        if pose[0] > self.pos[0] and self.left_pole < pose[1] < self.right_pole:
            return True
        else:
            return False
#        return self.radius > math.sqrt((self.pos[0]-pose[0])**2 + (self.pos[1]-pose[1])**2)

    def draw(self, ax, elems):
        x, y = self.pos
#        c = ax.scatter(x + 0.16, y + 0.5, s=50, marker=">", label="landmarks", color="red")
#        c = ax.scatter([x, x], [y - 1.0, y + 1.0], color="black")
#        elems.append(c)
        elems += ax.plot([x + 0.6, x + 0.6], [y - 1.3, y + 1.3], color="black")
        elems += ax.plot([x, x + 0.6], [y + 1.3, y + 1.3], color="black")
        elems += ax.plot([x, x + 0.6], [y - 1.3, y - 1.3], color="black")

class Ball:
    def __init__(self, pose, color="black"):
        self.pose = pose
        self.r = 0.1
        self.color = color
        self.collision_ball = False
        self.out_line = False

    def state_transition(self, time, pose, action):
        robot_x, robot_y, robot_theta = player.pose
#        enemy1_x, enemy1_y, enemy1_theta = enemy_player.pose
#        nu, omega = a.decision(action)
        nu, omega = action
        
        self.collision_ball = False
        if collision_detection_ball(player, ball):
            pose = pose + np.array([nu*math.cos(robot_theta), nu*math.sin(robot_theta)])*time
            self.collision_ball = True
        
#        if collision_detection_ball(enemy_player, ball):
#            pose = pose + np.array([nu*math.cos(enemy1_theta), nu*math.sin(enemy1_theta)])*time
            
        return pose

    def draw(self, ax, elems):
        x, y = self.pose
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)
        elems.append(ax.add_patch(c))

    def one_step(self, time_interval, action): 
        self.pose = self.state_transition(time_interval, self.pose, action)
#        elems.append(ax.text(-4.4, 4.5, "t= "+str(i), fontsize=10))
#        for obj in self.objects:
#            obj.draw(ax, elems)
 

#class Puddle:
#    def __init__(self, lowerleft, upperright, depth):
#        self.lowerleft = lowerleft
#        self.upperright = upperright
#        self.depth = depth
#
#    def draw(self, ax, elems):
#        w = self.upperright[0] - self.lowerleft[0]
#        h = self.upperright[1] - self.lowerleft[1]
#        r = patches.Rectangle(self.lowerleft, w, h, color="blue", alpha=self.depth)
#        elems.append(ax.add_patch(r))
#
#    def inside(self, pose):
#        return all([self.lowerleft[i] < pose[i] < self.upperright[i] for i in [0, 1]])
#
class SoccerWorld(World):
    def __init__(self, time_span, time_interval, debug=False):
        super().__init__(time_span, time_interval, debug)
#        self.puddles = []
        self.robots = []
        self.goals = []
#        self.ball = []

    def append(self, obj):
        self.objects.append(obj)
#        if isinstance(obj, Puddle): self.puddles.append(obj)
        if isinstance(obj, IdealRobot): self.robots.append(obj)
        if isinstance(obj, Goal): self.goals.append(obj)
#        if isinstance(obj, Ball): self.ball.append(obj)

#    def puddle_depth(self, pose):
#        return sum([p.depth*p.inside(pose) for p in self.puddles])
#
#    def one_step(self, i, elems, ax):
#        super().one_step(i, elems, ax)
#        for r in self.robots:
##            r.agent.puddle_depth = self.puddle_depth(r.pose)
#            for g in self.goals:
#                if g.inside(r.pose):
#                    r.agent.in_goal = True
#                    r.agent.final_value = g.value
#            if collision_detection_ball(r, ball) and r.agent.name == 'EnemyAgent':
#                a.cut_ball = True

    def one_step(self, action):
        super().one_step(action)
        for r in self.robots:
#            r.agent.puddle_depth = self.puddle_depth(r.pose)
            for g in self.goals:
                if g.inside(ball.pose):
                    r.agent.in_goal = True
                    r.agent.reward += 500
            if abs(ball.pose[0]) > 4.5 and abs(ball.pose[1]) > goal.right_pole:
                ball.out_line = True
                r.agent.reward += -450
            if abs(ball.pose[1]) > 3.0:
                ball.out_line = True
                r.agent.reward += -450
            if collision_detection_ball(r, ball) and r.agent.name == 'EnemyAgent':
                r.agent.cut_ball = True
            if abs(r.pose[0]) > 5.0 or abs(r.pose[1]) > 3.5: 
                r.agent.out_line = True
                r.agent.reward += -450
            poses = []
            poses.append(r.pose)
            poses.append(ball.pose)
            pose = list(itertools.chain.from_iterable(poses))
            reward = r.agent.reward
            if r.agent.in_goal or r.agent.out_line or ball.out_line:
                done = True
            else:
                done = False
        return pose, reward, done


class IdealRobot:
    def __init__(self, pose, agent=None, sensor=None, color="black"):
        self.pose = pose
        self.r = 0.1
        self.color = color
        self.agent = agent
        self.poses = [pose]
        self.sensor = sensor

    @classmethod
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10:
            return pose + np.array([nu*math.cos(t0), nu*math.sin(t0), omega])*time
        else:
            return pose + np.array([nu/omega*(math.sin(t0+omega*time)-math.sin(t0)), nu/omega*(-math.cos(t0+omega*time)+math.cos(t0)), omega*time])

    def draw(self, ax, elems):
        x,y,theta = self.pose
        xn = x + self.r*math.cos(theta)
        yn = y + self.r*math.sin(theta)
        elems += ax.plot([x,xn], [y,yn], color=self.color)
        #ax.plot([x,xn],[y,yn], color=self.color)
        c = patches.Circle(xy=(x,y), radius=self.r, fill=False, color=self.color)
        elems.append(ax.add_patch(c))
        #ax.add_patch(c)

        self.poses.append(self.pose)
#        elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color="black") #ロボットの軌跡
        if self.sensor and len(self.pose) > 1:
            self.sensor.draw(ax, elems, self.poses[-2])
        if self.agent and hasattr(self.agent, "draw"):
            self.agent.draw(ax, elems)

    def one_step(self, time_interval, action):
        if not self.agent: return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(action)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        return self.pose

    def choose_pose(self):
        t = random.random()*2*math.pi
#        y = random.random()*4-2
#        return np.array([-0.5, y, 0.0]).T, np.array([0.0, y]).T
        y = random.uniform(-2.5, 2.5)
        a = 4/9
        x = a*y**2
        ball_x = a*y**2 + 0.5
        return np.array([x, y, 0]).T, np.array([ball_x, y]).T

    def reset(self):
        self.pose, ball.pose = self.choose_pose()
        self.agent.in_goal = False
        self.agent.reward = 0.0
        self.agent.total_reward = 0.0
        ball.collision_ball = False
        self.agent.cut_ball = False
        self.agent.out_line = False
        ball.out_line = False
        world.time = 0
        poses = []
        poses.append(self.pose)
        poses.append(ball.pose)
        pose = list(itertools.chain.from_iterable(poses))
        return pose




class SoccerAgent(Agent):
    def __init__(self, time_interval, goal, obstacle_coef=10):
        super().__init__(0.0, 0.0)

        self.time_interval = time_interval
        self.obstacle_coef = obstacle_coef
#        self.puddle_depth = 0.0
        self.reward = 0.0
        self.total_reward = 0.0
        self.name = 'SoccerAgent'
        self.in_goal = False
        self.final_value = 0.0
        self.goal = goal
        self.cut_ball = False
        self.collision_ball = False
        self.out_line = False

    def set_reward(self):
        reward = 0
        x, y, theta = player.pose
        goal_dx, goal_dy = self.goal.pos[0] - x, self.goal.pos[1] - y
        goal_direction = int((math.atan2(goal_dy, goal_dx) - theta)*180/math.pi)
#        if goal_direction > 10: reward += -1
#        elif goal_direction < -10: reward += -1
#        else: reward += 1
#        goal_distance = distance_measurement(goal_dx, goal_dy)
#        reward += math.floor(-1.0 * goal_distance)
        ball_dx, ball_dy = ball.pose[0] - x, ball.pose[1] - y
        ball_distance = distance_measurement(ball_dx, ball_dy)
        reward += math.floor(-1.0 * ball_distance)
        ball_goal_distance = -1.0 * math.sqrt((self.goal.pos[0] - ball.pose[0])**2 + (self.goal.pos[1] - ball.pose[1])**2)
        reward += math.floor(ball_goal_distance)
#        if self.cut_ball: reward += -150
        return reward
#         return -1.0 - self.puddle_depth*self.puddle_coef
#        return -1.0 - self.obstacle_coef if self.cut_ball else -1.0 
    
#    @classmethod
#    def policy(cls, pose, goal):
#        x, y, theta = pose
#    
#        goal_dx, goal_dy = goal.pos[0] - x, goal.pos[1] - y
#        goal_direction = int((math.atan2(goal_dy, goal_dx) - theta)*180/math.pi)
#        goal_direction = (goal_direction + 360*1000 + 180)%360 - 180
#        ball_dx, ball_dy = ball.pose[0] - x, ball.pose[1] - y
#        ball_direction = int((math.atan2(ball_dy, ball_dx) - theta)*180/math.pi)
#        ball_direction = (ball_direction + 360*1000 + 180)%360 - 180
#    
#        if not collision_detection_ball(player, ball):
#            if ball_direction > 10: nu, omega = 0.0, 0.8
#            elif ball_direction < -10: nu, omega = 0.0, -0.8
#            else: nu, omega = 0.5, 0.0
#    
#        else:
#            if goal_direction > 10: nu, omega = 0.0, 0.8
#            elif goal_direction < -10: nu, omega = 0.0, -0.8
#            else: nu, omega = 0.5, 0.0
#    
#        return nu, omega

    
    def decision(self, action, observation=None):
        if self.in_goal:
            return 0.0, 0.0
        if self.cut_ball:
            return 0.0, 0.0
        if self.out_line:
            return 0.0, 0.0
    
    #    self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
    #    self.prev_nu, self.prev_omega = self.nu, self.omega
    #    self.estimator.observation_update(observation)
        self.reward = self.set_reward()
        self.total_reward += self.reward
#        self.total_reward += self.time_interval*self.reward_per_sec()
    
#        nu, omega = self.policy(player.pose, self.goal)
        nu, omega = action[0], action[1]
        self.prev_nu, self.prev_omega = nu, omega
        return nu, omega
    
    def draw(self, ax, elems):
        # super().draw(ax, elems)
        x, y, _ = player.pose
        elems.append(ax.text(x-2.0, y+0.5, "reward/sec:" + str(self.set_reward()), fontsize=8))
        elems.append(ax.text(x-2.0, y+1.0, "eval: {:.1f}".format(self.total_reward+self.final_value), fontsize=8))
            

class EnemyAgent(Agent):
    def __init__(self, time_interval, pose, goal, puddle_coef=100):
        super().__init__(0.0, 0.0)

        self.time_interval = time_interval
        self.pose = pose
        self.puddle_coef = puddle_coef
        self.puddle_depth = 0.0
        self.total_reward = 0.0
        self.name = 'EnemyAgent'
        self.in_goal = False
        self.final_value = 0.0
        self.goal = goal

    def reward_per_sec(self):
        return -1.0 - self.puddle_depth*self.puddle_coef

    @classmethod
    def policy(cls, pose, goal):
        x, y, theta = pose

        ball_dx, ball_dy = ball.pose[0] - x, ball.pose[1] - y
        ball_direction = int((math.atan2(ball_dy, ball_dx) - theta)*180/math.pi)
        ball_direction = (ball_direction + 360*1000 + 180)%360 - 180

        r = enemy_player.r + ball.r
        if (ball_dx*ball_dx + ball_dy*ball_dy) > (r*r):
            if ball_direction > 10: nu, omega = 0.0, 0.8
            elif ball_direction < -10: nu, omega = 0.0, -0.8
            else: nu, omega = 0.5, 0.0

        else:
            nu, omega = 0.0, 0.0
            
        return nu, omega

    def decision(self, observation=None):
        if self.in_goal:
            return 0.0, 0.0

#        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
#        self.prev_nu, self.prev_omega = self.nu, self.omega
#        self.estimator.observation_update(observation)

        self.total_reward += self.time_interval*self.reward_per_sec()
        nu, omega = self.policy(self.pose, self.goal)
        self.pose = IdealRobot.state_transition(nu, omega, time_interval, self.pose)
        self.prev_nu, self.prev_omega = nu, omega
        return nu, omega

    def draw(self, ax, elems):
        #super().draw(ax, elems)
        x, y, _ = self.pose
#        elems.append(ax.text(x+1.0, y-0.5, "reward/sec:" + str(self.reward_per_sec()), fontsize=8))
#        elems.append(ax.text(x+1.0, y-1.0, "eval: {:.1f}".format(self.total_reward+self.final_value), fontsize=8))
 

#class PuddleIgnoreAgent(EstimationAgent):
#    def __init__(self, time_interval, kf, goal, puddle_coef=100):
#        super().__init__(time_interval, 0.0, 0.0, kf)
#
#        self.puddle_coef = puddle_coef
#        self.puddle_depth = 0.0
#        self.total_reward = 0.0
#        self.in_goal = False
#        self.final_value = 0.0
#        self.goal = goal
#
#    def reward_per_sec(self):
#        return -1.0 - self.puddle_depth*self.puddle_coef
#
#    @classmethod
#    def policy(cls, pose, goal):
#        x, y, theta = pose
#        dx, dy = goal.pos[0] - x, goal.pos[1] - y
#        direction = int((math.atan2(dy, dx) - theta)*180/math.pi)
#        direction = (direction + 360*1000 + 180)%360 - 180
#
#        if direction > 10: nu, omega = 0.0, 2.0
#        elif direction < -10: nu, omega = 0.0, -2.0
#        else: nu, omega = 1.0, 0.0
#
#        return nu, omega
#
#    def decision(self, observation=None):
#        if self.in_goal:
#            return 0.0, 0.0
#
#        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
#        self.prev_nu, self.prev_omega = self.nu, self.omega
#        self.estimator.observation_update(observation)
#
#        self.total_reward += self.time_interval*self.reward_per_sec()
#
#        nu, omega = self.policy(self.estimator.pose, self.goal)
#        self.prev_nu, self.prev_omega = nu, omega
#        return nu, omega
#
#    def draw(self, ax, elems):
#        super().draw(ax, elems)
#        x, y, _ = self.estimator.pose
#        elems.append(ax.text(x+1.0, y-0.5, "reward/sec:" + str(self.reward_per_sec()), fontsize=8))
#        elems.append(ax.text(x+1.0, y-1.0, "eval: {:.1f}".format(self.total_reward+self.final_value), fontsize=8))
        
#if __name__ == '__main__':
time_interval = 0.1
world = SoccerWorld(30, time_interval, debug=False)

m = Map()
#    for ln in [(9, 2.6), (9, -2.6), (9, 6), (9, -6), (0, 6), (0, -6), (-9, 6), (-9, -6)]: m.append_landmark(Landmark(*ln))
#    world.append(m)

goal = Goal(4.5, 0)
world.append(goal)

#    world.append(Puddle((-2, 0), (0, 2), 0.1))
#    world.append(Puddle((-0.5, -2), (2.5, 1), 0.1))

init_pose = np.array([-0.5, 2.0, 0.0]).T
#    kf = KalmanFilter(m, initial_pose)
#    a = PuddleIgnoreAgent(time_interval, kf, goal)
#    r = Robot(initial_pose, sensor=Camera(m, distance_bias_rate_stddev=0, direction_bias_stddev=0), agent=a, color="red", bias_rate_stds=(0, 0))
a = SoccerAgent(time_interval, goal)
player = IdealRobot(init_pose, sensor=IdealCamera(m), agent=a, color="red")
world.append(player)
init_ball_pose = np.array([0.0, 2.0]).T
ball = Ball(init_ball_pose)
world.append(ball)
init_poses = []
#    for p in [[4, 1.5, math.pi], [3, -2, math.pi]]:
#        init_poses = np.array(p).T
#        enemy = EnemyAgent(time_interval, init_poses, ball.pose)
#        enemy_player = IdealRobot(init_poses, sensor=IdealCamera(m), agent=enemy, color="blue")
#        world.append(enemy_player)
world.draw()
#action = [0.8, 0.0]
#for i in range(10):
#    for num in range(100):
#        world.one_step(action)
#    player.reset()
 
