import pygame
import math
import numpy as np

pygame.init()
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("General Omniwheel Strategy Simulator")
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
    """
    Converts 4 omniwheel speeds into robot translation (vx, vy) and rotation.
    Simple + readable model, not physics-accurate but good for simulation.
    """
    vx = (m1 + m2 + m3 + m4) * 0.25
    vy = (-m1 + m2 + m3 - m4) * 0.25
    rot = (-m1 + m2 - m3 + m4) * 0.25
    return vx, vy, rot


# ------------------------------------------------------------
# MAIN LOGIC BLOCK (FLUID MOTION)
# ------------------------------------------------------------
def compute_motor_commands(robot_pos, robot_heading, ball_pos, enemies):
    robot_x, robot_y = robot_pos

    # FIELD PARAMS ---------------------------------------
    attract_strength = 2.0
    avoid_strength = 3.0
    swirl_strength = 20.0     # <-- this is what you were missing!!
    field_strength = 1.2
    rotation_gain = 5.0
    margin = 40

    force_x = 0
    force_y = 0

    # 1) MAIN ATTRACTION TO BALL -------------------------
    bx, by = ball_pos
    dx = bx - robot_x
    dy = by - robot_y
    dist = math.hypot(dx, dy)
    if dist > 1e-6:
        dx /= dist
        dy /= dist
    force_x += dx * attract_strength
    force_y += dy * attract_strength

    # 2) ENEMY REPULSION + SWIRL -------------------------
    for ex, ey in enemies:
        dx = robot_x - ex
        dy = robot_y - ey
        d = math.hypot(dx, dy)
        if d < 1e-6:
            continue

        # radial repulsion
        rep = avoid_strength / max(d, 1)
        force_x += (dx/d) * rep
        force_y += (dy/d) * rep

        # tangential swirl (90° rotated vector)
        tx = -dy / d
        ty =  dx / d

        swirl = swirl_strength / max(d, 1)
        force_x += tx * swirl
        force_y += ty * swirl

    # 3) WALL FORCE --------------------------------------
    field_w, field_h = field_size
    if robot_x < margin:
        force_x += field_strength
    if robot_x > field_w - margin:
        force_x -= field_strength
    if robot_y < margin:
        force_y += field_strength
    if robot_y > field_h - margin:
        force_y -= field_strength

    # 4) ROTATION TO FACE BALL ---------------------------
    aim_x = ball_pos[0] - robot_x
    aim_y = ball_pos[1] - robot_y
    target_angle = math.atan2(aim_y, aim_x)
    angle_error = wrap_angle(target_angle - robot_heading)
    rot = angle_error * rotation_gain

    # 5) OMNIWHEEL KINEMATICS ----------------------------
    vx = force_x
    vy = force_y

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

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # drag the ball/enemy
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
    if dragging_enemy is not None:
        enemies[dragging_enemy] = np.array(pygame.mouse.get_pos(), float)

    # run strategy
    m1, m2, m3, m4 = compute_motor_commands(robot_pos, robot_heading, ball_pos, enemies)

    # convert wheels -> motion
    vx, vy, rot = omni_to_velocity(m1, m2, m3, m4)
    robot_vel = np.array([vx, vy])

    robot_pos += robot_vel * 150 * dt
    robot_heading += rot * 2.0 * dt

    # --------------------------------------------------------
    # DRAW EVERYTHING
    # --------------------------------------------------------
    screen.fill((30, 30, 30))

    pygame.draw.circle(screen, (255, 145, 0), ball_pos.astype(int), BALL_RADIUS)

    for e in enemies:
        pygame.draw.circle(screen, (255, 50, 50), e.astype(int), ENEMY_RADIUS)

    pygame.draw.circle(screen, (100, 200, 255), robot_pos.astype(int), ROBOT_RADIUS)

    # heading line
    hx = robot_pos[0] + math.cos(robot_heading) * ROBOT_RADIUS
    hy = robot_pos[1] + math.sin(robot_heading) * ROBOT_RADIUS
    pygame.draw.line(screen, (200, 200, 255), robot_pos, (hx, hy), 3)

    font = pygame.font.SysFont("Arial", 18)
    txt = font.render(f"m1={m1:.2f}  m2={m2:.2f}  m3={m3:.2f}  m4={m4:.2f}", True, (220, 220, 220))
    screen.blit(txt, (10, 10))

    pygame.display.flip()

pygame.quit()