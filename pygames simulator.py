import pygame
import math
import numpy as np

pygame.init()
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Omniwheel Strategy Simulator with Ball Physics")
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
    """Normalise to [-pi, +pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def omni_to_velocity(m1, m2, m3, m4):
    """Convert 4 omniwheel speeds into vx, vy, rotation."""
    vx = (m1 + m2 + m3 + m4) * 0.25
    vy = (-m1 + m2 + m3 - m4) * 0.25
    rot = (-m1 + m2 - m3 + m4) * 0.25
    return vx, vy, rot


def compute_motor_commands(robot_pos, robot_heading, ball_pos, enemies):
    # ------------------------
    # TUNABLE CONSTANTS
    # ------------------------
    attract_strength = 1.5
    avoid_strength   = 10
    border_strength  = 2.0
    wrap_strength    = 2.0
    heading_keep_gain = 2.0
    BAD = 0.5  # misalignment threshold

    to_ball = ball_pos - robot_pos
    d = np.linalg.norm(to_ball)
    if d > 1e-6:
        to_ball /= d

    DESIRED_HEADING = 0.0
    GOAL_DIR = np.array([0.0, -1.0])

    alignment = to_ball @ GOAL_DIR

    robot_x, robot_y = robot_pos
    field_w, field_h = field_size
    force_x = 0.0
    force_y = 0.0

    # ------------------------
    # Behind-ball target
    # ------------------------
    OFFSET = 50 if alignment < 0.95 else 0
    shadow_point = ball_pos - GOAL_DIR * OFFSET
    to_target = shadow_point - robot_pos
    d_target = np.linalg.norm(to_target)
    if d_target > 1e-6:
        to_target /= d_target
        force_x += to_target[0] * attract_strength
        force_y += to_target[1] * attract_strength

    # ------------------------
    # Conditional swirl
    # ------------------------
    if d_target > 1e-6:
        swirl_dir = np.array([-to_ball[1], to_ball[0]]) if ball_pos[1] - robot_pos[1] > 0 else np.array([to_ball[1], -to_ball[0]])
        if alignment < BAD:
            t = np.clip((BAD - alignment) / (BAD + 1.0), 0, 1)
            wrap_scale = math.exp(-(d_target - 0.3) * 0.01)
            swirl_force = wrap_strength * t * wrap_scale
            force_x += swirl_dir[0] * swirl_force
            force_y += swirl_dir[1] * swirl_force

    # ------------------------
    # Enemy push-away
    # ------------------------
    for ex, ey in enemies:
        dx = robot_x - ex
        dy = robot_y - ey
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue
        strength = avoid_strength / max(dist, 1)
        dx /= dist
        dy /= dist
        force_x += dx * strength
        force_y += dy * strength

    # ------------------------
    # Border force
    # ------------------------
    margin = 40
    if robot_x < margin: force_x += border_strength
    if robot_x > field_w - margin: force_x -= border_strength
    if robot_y < margin: force_y += border_strength
    if robot_y > field_h - margin: force_y -= border_strength

    # ------------------------
    # Rotation
    # ------------------------
    heading_error = wrap_angle(DESIRED_HEADING - robot_heading)
    rot = heading_error * heading_keep_gain

    # ------------------------
    # Convert to wheel speeds
    # ------------------------
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

    # ------------------------
    # Events & dragging
    # ------------------------
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

    # ------------------------
    # Strategy -> motor commands
    # ------------------------
    m1, m2, m3, m4 = compute_motor_commands(robot_pos, robot_heading, ball_pos, enemies)
    vx, vy, rot = omni_to_velocity(m1, m2, m3, m4)
    robot_vel = np.array([vx, vy])
    robot_pos += robot_vel * 150 * dt
    robot_heading += rot * 2.0 * dt

    # ------------------------
    # Ball physics
    # ------------------------
    if not dragging_ball:
        # push from robot if touching
        to_ball = ball_pos - robot_pos
        dist = np.linalg.norm(to_ball)
        if dist < ROBOT_RADIUS + BALL_RADIUS and dist > 1e-6:
            push_strength = 2000.0
            ball_vel += (to_ball / dist) * push_strength * dt

        # update position
        ball_pos += ball_vel * dt
        ball_vel *= ball_friction

        # limit speed
        speed = np.linalg.norm(ball_vel)
        if speed > ball_max_speed:
            ball_vel = (ball_vel / speed) * ball_max_speed

        # border collision
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

    # ------------------------
    # DRAW EVERYTHING
    # ------------------------
    screen.fill((30, 30, 30))

    # ball
    pygame.draw.circle(screen, (255, 145, 0), ball_pos.astype(int), BALL_RADIUS)

    # enemies
    for e in enemies:
        pygame.draw.circle(screen, (255, 50, 50), e.astype(int), ENEMY_RADIUS)

    # robot
    pygame.draw.circle(screen, (100, 200, 255), robot_pos.astype(int), ROBOT_RADIUS)
    hx = robot_pos[0] + math.cos(robot_heading) * ROBOT_RADIUS
    hy = robot_pos[1] + math.sin(robot_heading) * ROBOT_RADIUS
    pygame.draw.line(screen, (200, 200, 255), robot_pos, (hx, hy), 3)

    # debug text
    font = pygame.font.SysFont("Arial", 18)
    txt = font.render(f"m1={m1:.2f}  m2={m2:.2f}  m3={m3:.2f}  m4={m4:.2f}", True, (220, 220, 220))
    screen.blit(txt, (10, 10))

    pygame.display.flip()

pygame.quit()