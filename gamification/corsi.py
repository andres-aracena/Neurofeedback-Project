# gamification/corsi.py
import pygame
import random
from gamification.base_game import BaseGame
from gamification.backgrounds import StarField


# ======================================================
# Clase principal del juego Corsi
# ======================================================
class CorsiGame(BaseGame):
    def __init__(self, grid_size=3):
        super().__init__(title="Corsi")
        self.grid_size = grid_size

        # ----- Estética -----
        self.block_size = 60
        self.spacing = 20
        self.background = StarField(
            self.screen.get_width(),
            self.screen.get_height(),
            num_stars=120,
            speed_range=(2, 6),
            enabled=False  # activar True si quieres estrellas
        )

        # ----- Estado -----
        self.blocks = []
        self.sequence = []
        self.user_sequence = []
        self.state = "intro"

        # ----- Timers -----
        self.show_time = 600
        self.gap_time = 250
        self.last_flash_time = 0
        self.show_index = 0
        self.show_flash_on = True
        self.start_delay = 1000
        self.delay_start_time = 0

        self.glow_duration = 350
        self.glow_timers = []

        # ----- Feedback -----
        self.points = 0
        self.level = 1
        self.max_level = 5  # 🚀 termina en nivel 5
        self.feedback_color = None
        self.feedback_end_time = 0

        # ----- Barra de progreso -----
        self.progress_value = 0.0
        self.progress_target = 0.0

        # ----- Neurofeedback -----
        self.brain_ratio = 1.0

        # ----- Fuentes -----
        try:
            self.font_big = pygame.font.SysFont("Orbitron", 50)
            self.font_mid = pygame.font.SysFont("Orbitron", 36)
            self.font_small = pygame.font.SysFont("Orbitron", 28)
        except Exception:
            self.font_big = pygame.font.SysFont(None, 50)
            self.font_mid = pygame.font.SysFont(None, 36)
            self.font_small = pygame.font.SysFont(None, 28)

        # Crear bloques centrados
        self.create_blocks()

    # ------------------------------------------------------
    def set_brain_ratio(self, ratio: float):
        self.brain_ratio = ratio

    # ======================================================
    # Crear bloques
    # ======================================================
    def create_blocks(self):
        w, h = self.screen.get_size()
        n = self.grid_size
        total_w = n * self.block_size + (n - 1) * self.spacing
        total_h = n * self.block_size + (n - 1) * self.spacing
        start_x = (w - total_w) // 2
        start_y = (h - total_h) // 2 - 10

        self.blocks = []
        for r in range(n):
            for c in range(n):
                x = start_x + c * (self.block_size + self.spacing)
                y = start_y + r * (self.block_size + self.spacing)
                self.blocks.append(pygame.Rect(x, y, self.block_size, self.block_size))
        self.glow_timers = [0] * len(self.blocks)

    # ======================================================
    # Generar secuencia dinámica según nivel
    # ======================================================
    def generate_sequence(self):
        length = min(2 + (self.level - 1), len(self.blocks))  # 2 en nivel 1, +1 por nivel
        self.sequence = random.sample(range(len(self.blocks)), length)
        self.user_sequence = []
        self.show_index = 0
        self.show_flash_on = True
        self.delay_start_time = pygame.time.get_ticks()
        self.state = "delay_before_sequence"

    # ======================================================
    # Eventos
    # ======================================================
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if self.state == "intro" and event.type == pygame.KEYDOWN:
                self.generate_sequence()

            elif self.state in ["game_over", "game_won"] and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.points, self.level = 0, 1
                    self.generate_sequence()

            elif self.state == "user_input" and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for idx, rect in enumerate(self.blocks):
                    if rect.collidepoint(event.pos):
                        now = pygame.time.get_ticks()
                        self.glow_timers[idx] = now + self.glow_duration
                        if len(self.user_sequence) < len(self.sequence):
                            self.user_sequence.append(idx)
                            self.progress_target = len(self.user_sequence) / len(self.sequence)
                        if len(self.user_sequence) >= len(self.sequence):
                            self.state = "verify"

    # ======================================================
    def update(self):
        now = pygame.time.get_ticks()

        if self.state == "delay_before_sequence":
            if now - self.delay_start_time >= self.start_delay:
                self.last_flash_time = now
                self.state = "show_sequence"

        elif self.state == "show_sequence":
            elapsed = now - self.last_flash_time
            if self.show_flash_on and elapsed >= self.show_time:
                self.show_flash_on = False
                self.last_flash_time = now
            elif not self.show_flash_on and elapsed >= self.gap_time:
                self.show_index += 1
                self.show_flash_on = True
                self.last_flash_time = now
                if self.show_index >= len(self.sequence):
                    self.state = "user_input"
                    self.user_sequence = []
                    self.progress_value = 0.0
                    self.progress_target = 0.0

        elif self.state == "verify":
            if self.feedback_end_time != 0 and now >= self.feedback_end_time:
                if self.level > self.max_level:
                    # Fin del juego -> evaluar resultado
                    if self.points >= 55:
                        self.state = "game_won"
                    else:
                        self.state = "game_over"
                else:
                    self.generate_sequence()
                self.feedback_end_time = 0
                self.feedback_color = None

        self.progress_value += (self.progress_target - self.progress_value) * 0.15

    # ======================================================
    def _draw_text(self, text, font, color, x, y, center=True, outline=True):
        if outline:
            for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
                shadow = font.render(text, True, (0,0,0))
                rect = shadow.get_rect(center=(x+dx,y+dy))
                self.screen.blit(shadow, rect)
        surface = font.render(text, True, color)
        rect = surface.get_rect(center=(x,y))
        self.screen.blit(surface, rect)

    def _draw_glow(self, rect, color=(0, 240, 255), alpha=80, inflate=20):
        s = pygame.Surface((rect.width + inflate*2, rect.height + inflate*2), pygame.SRCALPHA)
        pygame.draw.rect(s, (*color, alpha), s.get_rect(), border_radius=20)
        self.screen.blit(s, (rect.x - inflate, rect.y - inflate))

    def _draw_brain_ratio_bar(self, x, y, w, h):
        ratio_norm = max(0.0, min(1.0, self.brain_ratio / 5.0))
        pygame.draw.rect(self.screen, (40,40,50), (x, y, w, h), border_radius=10)
        pygame.draw.rect(self.screen, (0,200,255), (x, y + h - int(h*ratio_norm), w, int(h*ratio_norm)), border_radius=10)
        self._draw_text("Neurofeedback", self.font_small, (220,220,220), x + w//2, y - 20)
        self._draw_text(f"{self.brain_ratio:.2f}", self.font_small, (0,255,180), x + w//2, y + h + 20)

    # ======================================================
    def draw(self):
        self.background.run_frame(self.screen)
        w, h = self.screen.get_size()

        # Intro
        if self.state == "intro":
            self._draw_text("Bloques de Corsi", self.font_big, (0,255,200), w//2, 100)
            for i, line in enumerate([
                "Eres un explorador espacial.",
                "Restaura tu nave activando los bloques en orden.",
                "Presiona cualquier tecla para comenzar..."
            ]):
                self._draw_text(line, self.font_small, (200,200,200), w//2, 220 + i*40)
            return

        # Game Over / Win
        if self.state in ["game_over", "game_won"]:
            if self.state == "game_won":
                title, color, msg = "¡Energía Restaurada!", (0,255,136), "La nave vuelve a la vida."
            else:
                title, color, msg = "Misión Fallida", (255,80,80), "La nave quedó sin energía..."
            self._draw_text(title, self.font_big, color, w//2, h//2 - 40)
            self._draw_text(msg, self.font_mid, (220,220,220), w//2, h//2 + 10)
            self._draw_text("Presiona R para reiniciar", self.font_small, (0,255,200), w//2, h//2 + 60)
            return

        # Bloques
        now = pygame.time.get_ticks()
        seq_highlight_idx = self.sequence[self.show_index] if (self.state == "show_sequence" and self.show_flash_on and self.show_index < len(self.sequence)) else None

        for idx, rect in enumerate(self.blocks):
            color = (30, 42, 56)
            if seq_highlight_idx == idx:
                self._draw_glow(rect, (0,240,255), 110, 15)
                color = (0,240,255)
            elif idx in self.user_sequence:
                color = (100,255,200)
            pygame.draw.rect(self.screen, color, rect, border_radius=12)
            if self.glow_timers[idx] > now:
                self._draw_glow(rect, (0,255,180), 140, 18)

        # Barra progreso
        bar_x, bar_y, bar_w, bar_h = w//2 - 250, 40, 500, 26
        pygame.draw.rect(self.screen, (50,50,60), (bar_x, bar_y, bar_w, bar_h), border_radius=12)
        if self.state == "user_input":
            pygame.draw.rect(self.screen, (0,240,255), (bar_x, bar_y, int(bar_w*self.progress_value), bar_h), border_radius=12)
            self._draw_text(f"{len(self.user_sequence)} / {len(self.sequence)}", self.font_small, (200,200,200), w//2, bar_y+bar_h//2+2)
        else:
            msg = {"delay_before_sequence": "Prepárate...", "show_sequence": "Observa la secuencia...", "verify": "Verificando..."}.get(self.state, "")
            if msg:
                self._draw_text(msg, self.font_small, (200,200,200), w//2, bar_y+bar_h//2+2)

        # Puntos y nivel
        self._draw_text(f"{self.points} pts", self.font_big, (0,255,200), w//2, h-100)
        self._draw_text(f"Nivel {self.level}", self.font_mid, (220,220,220), w//2, h-50)

        # Barra lateral neurofeedback
        self._draw_brain_ratio_bar(w - 100, 100, 40, h - 200)

    # ======================================================
    def verify_sequence_and_prepare_feedback(self):
        now = pygame.time.get_ticks()
        if self.user_sequence == self.sequence:
            bonus = 10 + (5 if self.brain_ratio > 0.5 else 0)
            self.points += bonus
            self.level += 1
            self.feedback_color = (0,255,136)
        else:
            self.points = max(0, self.points - 5)
            self.level = max(1, self.level)  # no baja de 1
            self.feedback_color = (255,0,93)
        self.feedback_end_time = now + 900
        self.state = "verify"

    # ======================================================
    def run(self):
        while self.running:
            self.handle_events()
            if self.state == "verify" and self.feedback_end_time == 0:
                self.verify_sequence_and_prepare_feedback()
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()
        import sys; sys.exit()
