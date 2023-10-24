import itertools
import random
import time

import Box2D
from Box2D import b2Vec2
import numpy as np
import pygame
from world.car import Car
import map_processor


class World(Box2D.b2ContactListener):
    def __init__(self, num_cars):
        super().__init__()
        self.b2world = Box2D.b2World((0, 0))
        self.vehicles = []
        self.buildings = []
        self.edges = []
        self.b2world.contactListener = self
        self.zoom = 3

        self.rewards = [0] * num_cars

        self.num_cars = num_cars
        for i in range(num_cars):
            self.create_vehicle()

    def create_vehicle(self):
        self.vehicles.append(Car(self.b2world))
        return self.vehicles[-1]

    def make_building(self, x1, y1, x2, y2):
        buildDef = Box2D.b2BodyDef()
        buildDef.type = Box2D.b2_staticBody
        buildDef.position = Box2D.b2Vec2((x1 + x2) / 2, (y1 + y2) / 2)
        build = self.b2world.CreateBody(buildDef)
        buildShape = Box2D.b2PolygonShape()
        buildShape.SetAsBox((x2 - x1) / 2, (y2 - y1) / 2)
        buildFixtureDef = Box2D.b2FixtureDef()
        buildFixtureDef.shape = buildShape
        build.CreateFixture(buildFixtureDef)
        self.buildings.append(build)
        return build

    def add_edges(self, vertices, scale, shift_x, shift_y):
        if len(vertices) == 0:
            return
        if isinstance(vertices[0], list):
            vertices = itertools.chain(*vertices)
        for e in vertices:
            e0 = (e[0][0] * scale + shift_x, e[0][1] * scale + shift_y)
            e1 = (e[1][0] * scale + shift_x, e[1][1] * scale + shift_y)
            e = (e0, e1)

            buildDef = Box2D.b2BodyDef()
            buildDef.type = Box2D.b2_staticBody
            edge = self.b2world.CreateBody(buildDef)
            buildShape = Box2D.b2EdgeShape()
            buildShape.vertices = e
            buildFixtureDef = Box2D.b2FixtureDef()
            buildFixtureDef.shape = buildShape
            edge.CreateFixture(buildFixtureDef)
            self.edges.append(edge)

    def update(self, timestep):
        for veh in self.vehicles:
            veh.update_forces()

        self.b2world.Step(timestep, 8, 3)

        for veh in self.vehicles:
            if veh.dead:
                veh.destroy()
        self.vehicles = list(filter(lambda x: not x.dead, self.vehicles))

    def draw(self, window):
        canvas = pygame.Surface((window.get_width(), window.get_height()))

        scale = self.zoom
        if len(self.vehicles) > 0:
            x_shift = window.get_width() / 2 - self.vehicles[0].body.position.x * scale
            y_shift = window.get_height() / 2 - self.vehicles[0].body.position.y * scale
        else:
            x_shift = window.get_width() / 2
            y_shift = window.get_height() / 2

        drawn = []

        for item in self.vehicles:
            vertices = item.get_vertices()
            vertices = [(p.x * scale + x_shift, p.y * scale + y_shift) for p in vertices]
            drawn.append(pygame.draw.polygon(
                canvas,
                (50, 50, 255),
                vertices))
        for item in self.edges:
            vertices = item.fixtures[0].shape.vertices
            vertices = [(item.GetWorldPoint(b2Vec2(x, y))) for x, y in vertices]
            vertices = [(p.x * scale + x_shift, p.y * scale + y_shift) for p in vertices]
            if (0 <= vertices[0][0] <= window.get_width() and 0 <= vertices[0][1] <= window.get_height()) or \
                    (0 <= vertices[1][0] <= window.get_width() and 0 <= vertices[1][1] <= window.get_height()):
                drawn.append(pygame.draw.line(
                    canvas, (255, 0, 0),
                    (vertices[0][0], vertices[0][1]),
                    (vertices[1][0], vertices[1][1])))

        canvas = pygame.transform.flip(canvas, False, True)
        window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

    def cast_ray(self, source, pos, dir, range):
        p1 = b2Vec2(*pos)
        p2 = b2Vec2(pos[0] + dir[0] * range, pos[1] + dir[1] * range)

        out = None

        class Callback(Box2D.b2RayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                nonlocal out
                if source is not None and fixture in source.fixtures:
                    return -1

                out = point
                return 0

        self.b2world.RayCast(Callback(), p1, p2)
        return out

    def get_vision(self, car, points):
        out = np.ones(points)
        pos = (car.body.position.x, car.body.position.y)
        for i in range(points):
            angle_shift = i / points * np.pi * 2
            cast = self.cast_ray(
                car.body,
                pos,
                (np.cos(car.body.angle + angle_shift), np.sin(car.body.angle + angle_shift)),
                100
            )
            if cast is None:
                dist = np.inf
            else:
                dist = (Box2D.b2Vec2(*cast) - car.body.position).length
            out[i] = 1 / (dist + 1)
        return out

    def BeginContact(self, contact):
        if contact.fixtureA.body.userData is not None:
            contact.fixtureA.body.userData.dead = True
        if contact.fixtureB.body.userData is not None:
            contact.fixtureB.body.userData.dead = True


def main():
    pygame.init()
    pygame.display.init()
    window = pygame.display.set_mode((640, 480))

    world = World(1)

    edges = map_processor.to_edges('../squiggle.png')
    vertices = map_processor.edges_img_to_vertices(edges)
    vertices = map_processor.cut_corners(vertices, 20)
    world.add_edges(vertices, 0.7, -1220 * 0.7, -1000 * 0.7)

    last_tick = time.time()
    start = time.time()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    world.zoom *= 1.2
                elif event.y < 0:
                    world.zoom /= 1.2

        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_a]:
            turnangle = np.pi / 8
        elif pressed[pygame.K_d]:
            turnangle = -np.pi / 8
        else:
            turnangle = 0

        if pressed[pygame.K_w]:
            throttle = 1
        elif pressed[pygame.K_s]:
            throttle = -1
        else:
            throttle = 0

        if pressed[pygame.K_LCTRL]:
            brake = 1
        else:
            brake = 0

        if len(world.vehicles) > 0:
            veh = world.vehicles[0]
            if throttle != veh.throttle:
                start = time.time()
            veh.throttle = throttle
            veh.steer_angle = turnangle
            veh.brake = brake

        now = time.time()
        world.update(now - last_tick)
        world.draw(window)
        last_tick = now


if __name__ == '__main__':
    main()
