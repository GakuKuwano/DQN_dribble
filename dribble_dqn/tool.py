import math

def collision_detection_ball(person, ball):
    x, y, theta = person.pose
    robot_r = 0.1
    ball_r = ball.r
    
    dr = ball_r + robot_r
    dx = x - ball.pose[0]
    dy = y - ball.pose[1]

    if(dx*dx + dy*dy) <= (dr*dr):
        return True
    else:
        return False

def distance_measurement(x, y):
    return math.sqrt(x*x + y*y)
