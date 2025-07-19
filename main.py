import pygame as pg
pg.init()
screen = pg.display.set_mode((800, 600))
clock = pg.time.Clock()

# Define the Trapezoid class
class Trapezoid:
    def __init__(self, x_offset, y_offset, color, speed_x, speed_y, lane=0):
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.color = color
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.lane = lane

    def draw(self, screen):
        if self.lane == 0:
            # Draw the trapezoid for lane 0
            pg.draw.polygon(screen, self.color, [
                (356 - self.x_offset, 0 + self.y_offset),
                (396, 0 + self.y_offset),
                (396, 50 + self.y_offset),
                (342 - self.x_offset, 50 + self.y_offset)
            ])
        elif self.lane == 1:
            # Draw the trapezoid for lane 1
            pg.draw.polygon(screen, self.color, [
                (404, 0 + self.y_offset),
                (446 + self.x_offset, 0 + self.y_offset),
                (460 + self.x_offset, 50 + self.y_offset),
                (404, 50 + self.y_offset)
            ])
        # Update the position of the trapezoid
        self.x_offset += self.speed_x
        self.y_offset += self.speed_y

running = True

delay = 0

while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    screen.fill((0, 0, 0)) 

    pg.draw.line(screen, (0, 255, 0), (400, 0), (400, 600), 5)
    pg.draw.line(screen, (0, 255, 0), (350, 0), (200, 600), 5)
    pg.draw.line(screen, (0, 255, 0), (450, 0), (600, 600), 5)

    # Create and update the trapezoid instance outside the loop
    if 'trapezoid' not in locals():
        trapezoid = Trapezoid(x_offset=0, y_offset=0, color=(255, 0, 0), speed_x=0.25, speed_y=1, lane=1)
    
    # Draw and update the trapezoid position
    trapezoid.draw(screen)

    delay += 1

    xinR = 0
    yinR = 0

    if delay > 150:
        if 'trapezoid2' not in locals():
            trapezoid2 = Trapezoid(x_offset=0, y_offset=0, color=(255, 0, 0), speed_x=0.25, speed_y=1, lane=0)
        trapezoid2.draw(screen)

    clock.tick(60)
    pg.display.flip()       # Update the display

pg.quit()

