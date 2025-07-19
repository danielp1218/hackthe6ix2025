import pygame as pg
pg.init()
screen = pg.display.set_mode((800, 600))
clock = pg.time.Clock()

# Define the Trapezoid class



class Trapezoid:

    def __init__(self, x_offset, y_offset, color, lane=0):
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.color = color
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

running = True

NOTE_SPEED = 2
LINE_SLOPE = 0.25

notes = [
    {'x': 0, 'color': (255, 0, 0), 'time':0, 'lane': 1},
    {'x': 0, 'color': (0, 255, 0), 'time':1, 'lane': 0},
    {'x': 0, 'color': (0, 0, 255), 'time':2, 'lane': 1}
]

active_notes = []

while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    screen.fill((0, 0, 0)) 

    pg.draw.line(screen, (0, 255, 0), (400, 0), (400, 600), 5)
    pg.draw.line(screen, (0, 255, 0), (350, 0), (200, 600), 5)
    pg.draw.line(screen, (0, 255, 0), (450, 0), (600, 600), 5)

    while len(notes) > 0 and notes[0]['time'] <= pg.time.get_ticks() / 1000:
        active_notes.append(Trapezoid(notes[0]['x'], 0, notes[0]['color'], notes[0]['lane']))
        print("popping note:", notes[0])
        notes.pop(0)

    for note in active_notes:
        note.draw(screen)
        print("drawing note at x_offset:", note.x_offset, "y_offset:", note.y_offset, "lane:", note.lane)
        note.x_offset += LINE_SLOPE * NOTE_SPEED
        note.y_offset += NOTE_SPEED
        
    pg.display.flip()
    clock.tick(60)

pg.quit()

