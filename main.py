import pygame as pg

pg.init()
screen = pg.display.set_mode((800, 600))
clock = pg.time.Clock()

running = True

bullets = [[0, 50], [0, 100], [0, 200]]

while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    screen.fill((0, 0, 0)) 

    for b in bullets:
        b[0] += 1
        pg.draw.circle(screen, (255, 0, 0), (b[0], b[1]), 5)

    clock.tick(60)
    pg.display.flip()       # Update the display

pg.quit()

