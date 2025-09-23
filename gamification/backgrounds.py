# gamification/backgrounds.py
import pygame
import random

class StarField:
    def __init__(self, width, height, num_stars=40, speed_range=(2, 6), enabled=True):
        self.width = width
        self.height = height
        self.num_stars = num_stars
        self.speed_range = speed_range
        self.enabled = enabled  # <-- NUEVO
        self.stars = []
        self.center_x = width / 2
        self.center_y = height / 2
        self.background_color = (10, 10, 30)  # fondo oscuro tipo espacio

        if self.enabled:
            self.init_stars()

    def init_stars(self):
        """Genera estrellas distribuidas cerca del centro con velocidades aleatorias."""
        self.stars = []
        for _ in range(self.num_stars):
            offset_x = random.uniform(-200, 200)
            offset_y = random.uniform(-200, 200)
            star = {
                'x': self.center_x + offset_x,
                'y': self.center_y + offset_y,
                'speed': random.uniform(*self.speed_range),
                'color': (random.randint(200, 255), random.randint(200, 255), 255),
                'trail': []
            }
            self.stars.append(star)

    def update(self):
        if not self.enabled:
            return
        for star in self.stars:
            dx = star['x'] - self.center_x
            dy = star['y'] - self.center_y
            distance = max((dx**2 + dy**2)**0.5, 0.001)
            dx /= distance
            dy /= distance

            star['trail'].append((star['x'], star['y']))
            if len(star['trail']) > 15:
                star['trail'].pop(0)

            star['x'] += dx * star['speed']
            star['y'] += dy * star['speed']

            if not (0 <= star['x'] <= self.width and 0 <= star['y'] <= self.height):
                offset_x = random.uniform(-200, 200)
                offset_y = random.uniform(-200, 200)
                star['x'] = self.center_x + offset_x
                star['y'] = self.center_y + offset_y
                star['speed'] = random.uniform(*self.speed_range)
                star['color'] = (random.randint(200, 255), random.randint(200, 255), 255)
                star['trail'] = []

    def draw(self, screen):
        """Dibuja las estrellas y sus estelas en la pantalla."""
        screen.fill(self.background_color)

        if not self.enabled:
            return

        for star in self.stars:
            if len(star['trail']) > 1:
                for i in range(len(star['trail']) - 1):
                    alpha = int(255 * (i / len(star['trail'])))
                    color = (
                        min(star['color'][0] + alpha, 255),
                        min(star['color'][1] + alpha, 255),
                        min(star['color'][2] + alpha, 255)
                    )
                    pygame.draw.line(screen, color, star['trail'][i], star['trail'][i+1], 3)

            radius = max(2, int(star['speed']))
            pygame.draw.circle(screen, star['color'], (int(star['x']), int(star['y'])), radius)

    def run_frame(self, screen):
        """Actualizar y dibujar en un solo paso."""
        if not self.enabled:
            screen.fill(self.background_color)
            return
        self.update()
        self.draw(screen)
