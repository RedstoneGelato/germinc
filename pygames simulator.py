import pygame
import math
import numpy as np

pygame.init()
WIDTH, HEIGHT = 600, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Omniwheel Strategy Simulator (Angle+Distance Input)")
clock = pygame.time.Clock()

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------
field_size = (WIDTH, HEIGHT)

ROBOT_RADIUS = 18
ENEMY_RADIUS = 20
BALL_RADIUS = 10

# ------------------------------------------------------------
# WORLD OBJECTS
# ------------------------------------------------------------
robot_pos = np.array([WIDTH/2, HEIGHT/2], float)
robot_heading = 0.0
robot_vel = np.array([0.0, 0.0], float)

ball_pos = np.array([WIDTH/2 + 150, HEIGHT/2], float)
ball_vel = np.array([0.0, 0.0], float)
ball_mass = 1.0
ball_friction = 0.98
ball_max_speed = 600.0  # pixels/sec

enemies = [
    np.array([300, 250], float),
    np.array([600, 320], float)
]

dragging_ball = False
dragging_enemy = None


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def omni_to_velocity(m1, m2, m3, m4):
    vx = (m1 + m2 + m3 + m4) * 0.25
    vy = (-m1 + m2 + m3 - m4) * 0.25
    rot = (-m1 + m2 - m3 + m4) * 0.25
    return vx, vy, rot


# ------------------------------------------------------------
#      NEW STRATEGY: BALL GIVEN AS ANGLE + DISTANCE
# ------------------------------------------------------------
def compute_motor_commands(robot_pos, robot_heading, ball_dist, ball_angle, enemies):

    # Constants (same as before)
    attract_strength = 1.5
    avoid_strength   = 20
    border_strength  = 2.0
    wrap_strength    = 2.0
    heading_keep_gain = 2.0
    BAD = 0.3

    # Goal direction straight up
    GOAL_DIR = np.array([0.0, -1.0])
    DESIRED_HEADING = 3 * math.pi / 2

    # Convert polar → local vector
    to_ball_local = np.array([
        math.cos(ball_angle),
        math.sin(ball_angle)
    ])

    # Convert local → world
    R = np.array([
        [math.cos(robot_heading), -math.sin(robot_heading)],
        [math.sin(robot_heading),  math.cos(robot_heading)]
    ])
    to_ball_world = R @ to_ball_local

    alignment = to_ball_world @ GOAL_DIR

    robot_x, robot_y = robot_pos
    field_w, field_h = field_size

    force_x = 0.0
    force_y = 0.0

    # --------------------------------------------------------
    # BEHIND BALL TARGET (using angle+distance to reconstruct)
    # --------------------------------------------------------
    ball_world = robot_pos + to_ball_world * ball_dist
    OFFSET = 50 if alignment < 0.95 else 0
    shadow_point = ball_world - GOAL_DIR * OFFSET

    to_target = shadow_point - robot_pos
    dist_target = np.linalg.norm(to_target)

    if dist_target > 1e-6:
        to_target /= dist_target
        force_x += to_target[0] * attract_strength
        force_y += to_target[1] * attract_strength

    # --------------------------------------------------------
    # CONDITIONAL SWIRL (same behaviour as before)
    # --------------------------------------------------------
    if dist_target > 1e-6:
        # perpendicular swirl direction
        if ball_world[0] - robot_pos[0] > 0:
            swirl = np.array([-to_ball_world[1], to_ball_world[0]])
        else:
            swirl = np.array([ to_ball_world[1], -to_ball_world[0] ])

        if alignment < BAD:
            t = np.clip((BAD - alignment) / (BAD + 1.0), 0, 1)
            wrap_scale = math.exp(-(dist_target + 0.1) * 0.01)
            swirl_force = wrap_strength * t * wrap_scale
            force_x += swirl[0] * swirl_force
            force_y += swirl[1] * swirl_force

    # --------------------------------------------------------
    # ENEMY AVOIDANCE
    # --------------------------------------------------------
    for ex, ey in enemies:
        dx = robot_x - ex
        dy = robot_y - ey
        d = math.hypot(dx, dy)
        if d < 1e-6:
            continue
        push = avoid_strength / max(d, 1)
        dx /= d
        dy /= d
        force_x += dx * push
        force_y += dy * push

    # --------------------------------------------------------
    # BORDER REPULSION
    # --------------------------------------------------------
    margin = 40
    if robot_x < margin: force_x += border_strength
    if robot_x > field_w - margin: force_x -= border_strength
    if robot_y < margin: force_y += border_strength
    if robot_y > field_h - margin: force_y -= border_strength

    # --------------------------------------------------------
    # ROTATION CONTROL
    # --------------------------------------------------------
    heading_error = wrap_angle(DESIRED_HEADING - robot_heading)
    rot = heading_error * heading_keep_gain

    # --------------------------------------------------------
    # WHEEL OUTPUTS
    # --------------------------------------------------------
    vx, vy = force_x, force_y

    m1 =  vx - vy - rot
    m2 =  vx + vy + rot
    m3 =  vx + vy - rot
    m4 =  vx - vy + rot

    max_mag = max(abs(m1), abs(m2), abs(m3), abs(m4), 1)
    return m1/max_mag, m2/max_mag, m3/max_mag, m4/max_mag


# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
running = True
while running:
    dt = clock.tick(60) / 1000

    # Dragging events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if np.linalg.norm(ball_pos - [mx, my]) < BALL_RADIUS:
                dragging_ball = True
            for i, e in enumerate(enemies):
                if np.linalg.norm(e - [mx, my]) < ENEMY_RADIUS:
                    dragging_enemy = i
        if event.type == pygame.MOUSEBUTTONUP:
            dragging_ball = False
            dragging_enemy = None

    if dragging_ball:
        ball_pos = np.array(pygame.mouse.get_pos(), float)
        ball_vel[:] = 0
    if dragging_enemy is not None:
        enemies[dragging_enemy] = np.array(pygame.mouse.get_pos(), float)

    # --------------------------------------------------------
    # SENSOR → ANGLE + DIST
    # --------------------------------------------------------
    vec = ball_pos - robot_pos
    ball_dist = np.linalg.norm(vec)
    if ball_dist < 1e-6:
        ball_dist = 1e-6
    world_angle = math.atan2(vec[1], vec[0])
    ball_angle = wrap_angle(world_angle - robot_heading)

    # --------------------------------------------------------
    # STRATEGY (now only using angle+distance)
    # --------------------------------------------------------
    m1, m2, m3, m4 = compute_motor_commands(
        robot_pos, robot_heading, ball_dist, ball_angle, enemies
    )

    vx, vy, rot = omni_to_velocity(m1, m2, m3, m4)
    robot_vel = np.array([vx, vy])
    robot_pos += robot_vel * 150 * dt
    robot_heading += rot * 2.0 * dt

    # --------------------------------------------------------
    # BALL PHYSICS
    # --------------------------------------------------------
    if not dragging_ball:
        to_ball = ball_pos - robot_pos
        d = np.linalg.norm(to_ball)
        if d < ROBOT_RADIUS + BALL_RADIUS and d > 1e-6:
            push_strength = 2000.0
            ball_vel += (to_ball / d) * push_strength * dt

        ball_pos += ball_vel * dt
        ball_vel *= ball_friction

        speed = np.linalg.norm(ball_vel)
        if speed > ball_max_speed:
            ball_vel = (ball_vel / speed) * ball_max_speed

        if ball_pos[0] < BALL_RADIUS:
            ball_pos[0] = BALL_RADIUS
            ball_vel[0] *= -0.5
        if ball_pos[0] > WIDTH - BALL_RADIUS:
            ball_pos[0] = WIDTH - BALL_RADIUS
            ball_vel[0] *= -0.5
        if ball_pos[1] < BALL_RADIUS:
            ball_pos[1] = BALL_RADIUS
            ball_vel[1] *= -0.5
        if ball_pos[1] > HEIGHT - BALL_RADIUS:
            ball_pos[1] = HEIGHT - BALL_RADIUS
            ball_vel[1] *= -0.5

    # --------------------------------------------------------
    # DRAW EVERYTHING
    # --------------------------------------------------------
    screen.fill((30, 30, 30))

    pygame.draw.circle(screen, (255, 145, 0), ball_pos.astype(int), BALL_RADIUS)

    for e in enemies:
        pygame.draw.circle(screen, (255, 50, 50), e.astype(int), ENEMY_RADIUS)

    pygame.draw.circle(screen, (100, 200, 255), robot_pos.astype(int), ROBOT_RADIUS)
    hx = robot_pos[0] + math.cos(robot_heading) * ROBOT_RADIUS
    hy = robot_pos[1] + math.sin(robot_heading) * ROBOT_RADIUS
    pygame.draw.line(screen, (200, 200, 255), robot_pos, (hx, hy), 3)

    font = pygame.font.SysFont("Arial", 18)
    txt = font.render(f"BallDist={ball_dist:.1f}  BallAngle={math.degrees(ball_angle):.1f}", True, (220, 220, 220))
    screen.blit(txt, (10, 10))

    pygame.display.flip()

pygame.quit()