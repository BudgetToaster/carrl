import random
import time

import Box2D
from Box2D import b2Vec2
import numpy as np
import pyglet
from pyglet import shapes
from world.car import Car


class World(Box2D.b2ContactListener):
    def __init__(self, num_cars):
        super().__init__()
        self.b2world = Box2D.b2World((0, 0))
        self.vehicles = []
        self.obstacles = []
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
        self.obstacles.append(build)
        return build

    def update(self, timestep):
        for veh in self.vehicles:
            veh.update_forces()

        self.b2world.Step(timestep, 8, 3)

        for veh in self.vehicles:
            if veh.dead:
                veh.destroy()
        self.vehicles = list(filter(lambda x: not x.dead, self.vehicles))

    def draw(self, window):
        batch = pyglet.graphics.Batch()

        scale = self.zoom
        if len(self.vehicles) > 0:
            x_shift = window.width / 2 - self.vehicles[0].body.position.x * scale
            y_shift = window.height / 2 - self.vehicles[0].body.position.y * scale
        else:
            x_shift = window.width / 2
            y_shift = window.height / 2

        drawn = []

        for item in self.vehicles:
            vertices = item.get_vertices()
            vertices = [(p.x * scale + x_shift, p.y * scale + y_shift) for p in vertices]
            drawn.append(pyglet.shapes.Polygon(
                *vertices,
                color=(50, 50, 255), batch=batch))
        for item in self.obstacles:
            vertices = item.fixtures[0].shape.vertices
            vertices = [(item.GetWorldPoint(b2Vec2(x, y))) for x, y in vertices]
            vertices = [(p.x * scale + x_shift, p.y * scale + y_shift) for p in vertices]
            drawn.append(pyglet.shapes.Polygon(
                *vertices,
                color=(255, 0, 0), batch=batch))

        batch.draw()

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
        out = np.ones(points) * float('inf')
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
                continue
            dist = (Box2D.b2Vec2(*cast) - car.body.position).length
            out[i] = 1 / (dist + 1)
        return out

    def BeginContact(self, contact):
        if contact.fixtureA.body.userData is not None:
            contact.fixtureA.body.userData.dead = True
        if contact.fixtureB.body.userData is not None:
            contact.fixtureB.body.userData.dead = True


def main():
    window = pyglet.window.Window(640, 480)
    keys = pyglet.window.key.KeyStateHandler()
    window.push_handlers(keys)

    world = World(1)
    n = 5
    s = 100
    for x in range(-n, n+1):
        shift = random.random() * 100 - 50
        for y in range(-n, n+1):
            lanes = 2
            ix = x * s
            iy = y * s
            world.make_building(ix + 3.65 / 2 * lanes,
                                iy + 3.65 / 2 * lanes + shift,
                                ix + s - 3.65 / 2 * lanes,
                                iy + s - 3.65 / 2 * lanes + shift)

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        if scroll_y > 0:
            world.zoom *= 1.2
        elif scroll_y < 0:
            world.zoom /= 1.2

    last_tick = time.time()
    start = time.time()
    @window.event
    def on_draw():
        nonlocal last_tick, start
        window.clear()
        world.draw(window)
        if keys[pyglet.window.key.A]:
            turnangle = np.pi / 8
        elif keys[pyglet.window.key.D]:
            turnangle = -np.pi / 8
        else:
            turnangle = 0

        if keys[pyglet.window.key.W]:
            throttle = 1
        elif keys[pyglet.window.key.S]:
            throttle = -1
        else:
            throttle = 0

        if keys[pyglet.window.key.LCTRL]:
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
        last_tick = now

    pyglet.app.run()


if __name__ == '__main__':
    main()
