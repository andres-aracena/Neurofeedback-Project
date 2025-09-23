import pygame
import random
import sys

pygame.init()

# Configuración
WIDTH, HEIGHT = 800, 600
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Corsi Blocks: Misión Galáctica")

FONT = pygame.font.SysFont("Arial", 28)
BIG_FONT = pygame.font.SysFont("Arial", 40, bold=True)

# Colores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLOCK_COLORS = [(0, 200, 255), (255, 150, 0), (0, 255, 150), (255, 50, 100)]

# Variables de juego
level = 1
sequence = []
user_sequence = []
blocks = []
block_size = 100
game_over = False
show_sequence = True
seq_index = 0
timer = 0
flash_time = 800  # ms

# Crear posiciones de bloques
for i in range(2):
    for j in range(2):
        x = WIDTH//2 - 150 + j*200
        y = HEIGHT//2 - 150 + i*200
        blocks.append(pygame.Rect(x, y, block_size, block_size))

def draw_text(text, y, big=False, color=WHITE):
    font = BIG_FONT if big else FONT
    render = font.render(text, True, color)
    rect = render.get_rect(center=(WIDTH//2, y))
    WINDOW.blit(render, rect)

def new_sequence():
    return [random.randint(0, len(blocks)-1) for _ in range(level+2)]

# Inicializar primera secuencia
sequence = new_sequence()

clock = pygame.time.Clock()

while True:
    WINDOW.fill(BLACK)

    if game_over:
        draw_text("¡Misión Fallida! La nave quedó sin energía.", HEIGHT//2 - 50, big=True, color=(255,0,0))
        draw_text("Presiona R para reiniciar", HEIGHT//2 + 20)
    else:
        draw_text(f"Nivel {level}: Restaura el sistema de la nave", 50)

        # Mostrar secuencia
        if show_sequence:
            now = pygame.time.get_ticks()
            if now - timer > flash_time:
                timer = now
                seq_index += 1
                if seq_index >= len(sequence):
                    show_sequence = False
                    seq_index = 0

        for i, block in enumerate(blocks):
            color = BLOCK_COLORS[i % len(BLOCK_COLORS)]
            if show_sequence and seq_index < len(sequence) and i == sequence[seq_index]:
                pygame.draw.rect(WINDOW, WHITE, block.inflate(20, 20))  # destello
            pygame.draw.rect(WINDOW, color, block)
            pygame.draw.rect(WINDOW, WHITE, block, 3)

    # Eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if game_over and event.key == pygame.K_r:
                level = 1
                sequence = new_sequence()
                user_sequence = []
                game_over = False
                show_sequence = True
                timer = 0
                seq_index = 0

        if event.type == pygame.MOUSEBUTTONDOWN and not show_sequence and not game_over:
            pos = pygame.mouse.get_pos()
            for i, block in enumerate(blocks):
                if block.collidepoint(pos):
                    user_sequence.append(i)
                    if user_sequence[-1] != sequence[len(user_sequence)-1]:
                        game_over = True
                    elif len(user_sequence) == len(sequence):
                        level += 1
                        sequence = new_sequence()
                        user_sequence = []
                        show_sequence = True
                        timer = pygame.time.get_ticks()
                        seq_index = 0

    pygame.display.flip()
    clock.tick(60)
