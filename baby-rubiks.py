import pygame
import sys

pygame.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 600, 600
GRID_SIZE = 3
TILE_SIZE = 100
GRID_ORIGIN = ((SCREEN_WIDTH - TILE_SIZE * GRID_SIZE) // 2, (SCREEN_HEIGHT - TILE_SIZE * GRID_SIZE) // 2)
BUTTON_SIZE = 40
BUTTON_MARGIN = 10

# Colors
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLACK = (0, 0, 0)

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Puzzle World")
font = pygame.font.SysFont(None, 48)
small_font = pygame.font.SysFont(None, 32)

# Initialize grid 1-9
grid = [[1 + r * GRID_SIZE + c for c in range(GRID_SIZE)] for r in range(GRID_SIZE)]

# Define shifts

def shift_row(r, direction):
    if direction == 'left':
        grid[r] = grid[r][1:] + [grid[r][0]]
    else:
        grid[r] = [grid[r][-1]] + grid[r][:-1]


def shift_col(c, direction):
    col = [grid[r][c] for r in range(GRID_SIZE)]
    if direction == 'up':
        col = col[1:] + [col[0]]
    else:
        col = [col[-1]] + col[:-1]
    for r in range(GRID_SIZE):
        grid[r][c] = col[r]

# Button definitions
buttons = []
# Up buttons above each column
for c in range(GRID_SIZE):
    x = GRID_ORIGIN[0] + c * TILE_SIZE + (TILE_SIZE - BUTTON_SIZE) // 2
    y = GRID_ORIGIN[1] - BUTTON_SIZE - BUTTON_MARGIN
    rect = pygame.Rect(x, y, BUTTON_SIZE, BUTTON_SIZE)
    buttons.append({'rect': rect, 'type': 'col', 'index': c, 'dir': 'up', 'label': '^'})
# Down buttons below each column
for c in range(GRID_SIZE):
    x = GRID_ORIGIN[0] + c * TILE_SIZE + (TILE_SIZE - BUTTON_SIZE) // 2
    y = GRID_ORIGIN[1] + GRID_SIZE * TILE_SIZE + BUTTON_MARGIN
    rect = pygame.Rect(x, y, BUTTON_SIZE, BUTTON_SIZE)
    buttons.append({'rect': rect, 'type': 'col', 'index': c, 'dir': 'down', 'label': 'v'})
# Left buttons left of each row
for r in range(GRID_SIZE):
    x = GRID_ORIGIN[0] - BUTTON_SIZE - BUTTON_MARGIN
    y = GRID_ORIGIN[1] + r * TILE_SIZE + (TILE_SIZE - BUTTON_SIZE) // 2
    rect = pygame.Rect(x, y, BUTTON_SIZE, BUTTON_SIZE)
    buttons.append({'rect': rect, 'type': 'row', 'index': r, 'dir': 'left', 'label': '<'})
# Right buttons right of each row
for r in range(GRID_SIZE):
    x = GRID_ORIGIN[0] + GRID_SIZE * TILE_SIZE + BUTTON_MARGIN
    y = GRID_ORIGIN[1] + r * TILE_SIZE + (TILE_SIZE - BUTTON_SIZE) // 2
    rect = pygame.Rect(x, y, BUTTON_SIZE, BUTTON_SIZE)
    buttons.append({'rect': rect, 'type': 'row', 'index': r, 'dir': 'right', 'label': '>'})

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            for btn in buttons:
                if btn['rect'].collidepoint(pos):
                    if btn['type'] == 'row':
                        shift_row(btn['index'], btn['dir'])
                    else:
                        shift_col(btn['index'], btn['dir'])

    # Draw
    screen.fill(WHITE)

    # Draw grid tiles
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            x = GRID_ORIGIN[0] + c * TILE_SIZE
            y = GRID_ORIGIN[1] + r * TILE_SIZE
            rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, GRAY, rect, 2)
            num_text = font.render(str(grid[r][c]), True, BLACK)
            text_rect = num_text.get_rect(center=rect.center)
            screen.blit(num_text, text_rect)

    # Draw buttons
    for btn in buttons:
        pygame.draw.rect(screen, GRAY, btn['rect'])
        label = small_font.render(btn['label'], True, BLACK)
        lbl_rect = label.get_rect(center=btn['rect'].center)
        screen.blit(label, lbl_rect)

    pygame.display.flip()

pygame.quit()
sys.exit()
