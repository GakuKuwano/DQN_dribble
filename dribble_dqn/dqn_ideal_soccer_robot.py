import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np
import matplotlib
#matplotlib.use('nbagg')
import matplotlib.animation as anm
import time

class World:
    def __init__(self, time_span, time_interval, debug=False):
        self.objects = []
        self.debug = debug
        self.time_span = time_span
        self.time_interval = time_interval
        self.time = 0

    def append(self,obj):
        self.objects.append(obj)

#    def draw(self):
#        fig = plt.figure(figsize=(5,4))
#        ax = fig.add_subplot(111)
#        ax.set_aspect('equal')
#        ax.set_xlim(-5.0, 5.0)
#        ax.set_ylim(-3.5, 3.5)
#        ax.set_xlabel("X", fontsize=10)
#        ax.set_ylabel("Y", fontsize=10)
#
#        elems = []
#
#        if self.debug:
#            for i in range(1000): self.one_step(i, elems, ax)
#        else:
#            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems,ax), frames=int(self.time_span/self.time_interval)+1, interval=int(self.time_interval*1000), repeat=False)
#            plt.show()
#
#    def one_step(self, i, elems, ax):
#        while elems: elems.pop().remove()
#        time_str = "t = %.2f[s]" % (self.time_interval*i)
#        elems.append(ax.text(-4.4, 4.5, "t = "+str(i), fontsize=10))
#        for obj in self.objects:
#            obj.draw(ax, elems)
#            if hasattr(obj, "one_step"): obj.one_step(self.time_interval)
    
    def draw(self):
        fig = plt.figure(figsize=(5,4)) #5, 4
        self.ax = fig.add_subplot(111)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-5.5, 5.5)
        self.ax.set_ylim(-3.5, 3.5)
        self.ax.set_xlabel("X", fontsize=10)
        self.ax.set_ylabel("Y", fontsize=10)
        self.ax.plot([-4.5, 4.5], [3.0, 3.0], color="black")
        self.ax.plot([-4.5, 4.5], [-3.0, -3.0], color="black")
        self.ax.plot([4.5, 4.5], [-3.0, 3.0], color="black")
        self.ax.plot([-4.5, -4.5], [-3.0, 3.0], color="black")
        self.elems = []

#        self.ax.set_aspect('equal')
#        self.ax.set_xlim(-5.0, 5.0)
#        self.ax.set_ylim(-3.5, 3.5)
#        self.ax.set_xlabel("X", fontsize=10)
#        self.ax.set_ylabel("Y", fontsize=10)
    
    def one_step(self, action):
#        if self.debug:
#            for i in range(1000): self.one_step(i, elems, ax)
#        else:
#            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems,ax), frames=int(self.time_span/self.time_interval)+1, interval=int(self.time_interval*1000), repeat=False)
        while self.elems: self.elems.pop().remove()
        time_str = "t = %.2f[s]" % (self.time_interval*self.time)
        self.elems.append(self.ax.text(-4.4, 4.5, "t = "+str(self.time), fontsize=10))
        for obj in self.objects:
            obj.draw(self.ax, self.elems)
            if hasattr(obj, "one_step"): obj.one_step(self.time_interval, action)
        self.time+=1
#        plt.show()
        plt.draw()
        plt.pause(.001)
#        plt.clf()
#        plt.close()


#class IdealRobot:
#    def __init__(self, pose, agent=None, sensor=None, color="black"):
#        self.pose = pose
#        self.r = 0.1
#        self.color = color
#        self.agent = agent
#        self.poses = [pose]
#        self.sensor = sensor
#
#    @classmethod
#    def state_transition(cls, nu, omega, time, pose):
#        t0 = pose[2]
#        if math.fabs(omega) < 1e-10:
#            return pose + np.array([nu*math.cos(t0), nu*math.sin(t0), omega])*time
#        else:
#            return pose + np.array([nu/omega*(math.sin(t0+omega*time)-math.sin(t0)), nu/omega*(-math.cos(t0+omega*time)+math.cos(t0)), omega*time])
#
#    def draw(self, ax, elems):
#        x,y,theta = self.pose
#        xn = x + self.r*math.cos(theta)
#        yn = y + self.r*math.sin(theta)
#        elems += ax.plot([x,xn], [y,yn], color=self.color)
#        #ax.plot([x,xn],[y,yn], color=self.color)
#        c = patches.Circle(xy=(x,y), radius=self.r, fill=False, color=self.color)
#        elems.append(ax.add_patch(c))
#        #ax.add_patch(c)
#
#        self.poses.append(self.pose)
##        elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color="black") #ロボットの軌跡
#        if self.sensor and len(self.pose) > 1:
#            self.sensor.draw(ax, elems, self.poses[-2])
#        if self.agent and hasattr(self.agent, "draw"):
#            self.agent.draw(ax, elems)
#
#    def one_step(self, time_interval):
#        if not self.agent: return
#        obs = self.sensor.data(self.pose) if self.sensor else None
#        nu, omega = self.agent.decision(obs)
#        self.pose = self.state_transition(nu, omega, time_interval, self.pose)


class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def decision(self, observation=None):
        return self.nu, self.omega

class Landmark:
    def __init__(self, x, y):
        self.pos = np.array([x, y]).T
        self.id = None

    def draw(self, ax, elems):
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="landmarks", color="orange")
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10))


class Map:
    def __init__(self):
        self.landmarks = []

    def append_landmark(self, landmark):
        landmark.id = len(self.landmarks) + 1
        self.landmarks.append(landmark)

    def draw(self, ax, elems):
        for lm in self.landmarks: lm.draw(ax, elems)

class IdealCamera:
    def __init__(self, env_map, \
            distance_range=(0.5, 6.0),
            direction_range=(-math.pi/3, math.pi/3)):
        self.map = env_map
        self.lastdata = []

        self.distance_range = distance_range
        self.direction_range = direction_range

    def visible(self, polarpos):
        if polarpos is None:
            return False

        return self.distance_range[0] <= polarpos[0] <= self.distance_range[1] \
                and self.direction_range[0] <= polarpos[1] <= self.direction_range[1]

    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            if self.visible(z):
                observed.append((z, lm.id))
        self.lastdata = observed
        return observed

    @classmethod
    def observation_function(cls, cam_pose, obj_pos):
        diff = obj_pos - cam_pose[0:2]
        phi = math.atan2(diff[1], diff[0]) - cam_pose[2]
        while phi >= np.pi: phi -= 2*np.pi
        while phi < -np.pi: phi += 2*np.pi
        return np.array([np.hypot(*diff), phi]).T

    def draw(self, ax, elems, cam_pose):
        for lm in self.lastdata:
            x, y, theta = cam_pose
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance * math.cos(direction + theta)
            ly = y + distance * math.sin(direction + theta)
            elems += ax.plot([x, lx], [y, ly], color='pink')


class EnemyIdealRobot:
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

    def one_step(self, time_interval):
        if not self.agent: return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)


#class Ball:
#    def __init__(self, pose, color="black"):
#        self.pose = pose
#        self.r = 0.2
#        self.color = color
#
#    def state_transition(cls, time, pose):
#        robot_x, robot_y, robot_theta = player.pose
#        nu = straight.nu
#        ball_x, ball_y = pose
#        
#        r = 0.4
#        a = robot_x - ball_x
#        b = robot_y - ball_y
#        
#        if(a*a + b*b) <= (r*r):
#            return pose + np.array([nu*math.cos(robot_theta), nu*math.sin(robot_theta)])*time
#        else:
#            return pose
#
#    def draw(self, ax, elems):
#        x, y = self.pose
#        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)
#        ax.add_patch(c)
#
#    def one_step(self, time_interval):
#        self.pose = self.state_transition(time_interval, self.pose)
#        elems.append(ax.text(-4.4, 4.5, "t= "+str(i), fontsize=10))
#        for obj in self.objects:
#            obj.draw(ax, elems)
#        pass

#class BallAgent:
#    def __init__(self)

if __name__ == '__main__':
    world = World(10, 0.1)
    m = Map()
    m.append_landmark(Landmark(2, -2))
    m.append_landmark(Landmark(-1, -3))
    m.append_landmark(Landmark(3, 3))
    world.append(m)
    straight = Agent(0.2, 0.0)
    circling = Agent(0.2, 20.0/180*math.pi)
    player = IdealRobot(np.array([2,3,math.pi/6]).T, sensor=IdealCamera(m), agent=straight)
    robot2 = IdealRobot(np.array([-2,-1,math.pi/5*6]).T, sensor=IdealCamera(m), agent=circling, color="red")
    world.append(player)
    world.append(robot2)
#    ball = Ball(np.array([0, 0]).T)
#    world.append(ball)
    world.draw()
    for num in range(100):
        world.one_step()
