import pygame
import sys

class BaseGame:
    def __init__(self, title="Neurofeedback Game", width=850, height=450, bg_color=(20, 20, 30)):
        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.bg_color = bg_color
        self.running = True

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        """Actualizar estado del juego (override en subclases)."""
        pass

    def draw(self):
        """Dibujar elementos (override en subclases)."""
        pass

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.screen.fill(self.bg_color)
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
        pygame.quit()
        sys.exit()
