import Box2D
import numpy as np


class Car:
    def __init__(self, b2world, fwd=True, rwd=False):
        super().__init__()
        assert fwd or rwd

        self.fwd, self.rwd = fwd, rwd

        self.width = 4.572
        self.height = 1.8
        self.theor_max_speed = 65 # m/s, max speed WITHOUT friction
        self.zero_to_60 = 7 # 0-60 in mph
        #self.acceleration_coeff = 26.8 / (self.zero_to_60 * self.theor_max_speed - 26.8 * self.zero_to_60)
        self.acceleration_coeff = -np.log(1 - 26.8/self.theor_max_speed)/self.zero_to_60

        self.b2world = b2world
        carDef = Box2D.b2BodyDef()
        carDef.type = Box2D.b2_dynamicBody
        carDef.angle = np.pi / 2
        self.body = self.b2world.CreateBody(carDef)
        carShape = Box2D.b2PolygonShape()
        carShape.SetAsBox(self.width / 2, self.height / 2)
        carFixtureDef = Box2D.b2FixtureDef()
        carFixtureDef.shape = carShape
        carFixtureDef.density = 1
        carFixtureDef.friction = 0
        self.body.CreateFixture(carFixtureDef)

        self.body.massData.center.x = self.width / 5

        self.body.userData = self
        self.dead = False
        self.brake = 0

        self.steer_angle = 0
        # between -1 and 1
        self.throttle = 0

    def update_forces(self):
        def apply_linear_friction(pos, dir, strength):
            linvel = self.body.GetLinearVelocityFromWorldPoint(pos)
            dotprod = dir.x * linvel.x + dir.y * linvel.y
            self.body.ApplyForce(-dir * dotprod * strength, pos, wake=True)

        def apply_lateral_friction(pos, dir, max_force):
            dir_ortho = Box2D.b2Vec2(dir.y, -dir.x)
            linvel = self.body.GetLinearVelocityFromWorldPoint(pos)
            dotprod = dir_ortho.x * linvel.x + dir_ortho.y * linvel.y
            smoothing_fac = 40 / max(1, max_force)
            dotprod = np.clip(dotprod * smoothing_fac, -1, 1)
            self.body.ApplyForce(-dir_ortho * dotprod * max_force, pos, wake=True)

        def apply_brakes(pos, dir, max_force):
            linvel = self.body.GetLinearVelocityFromWorldPoint(pos)
            dotprod = dir.x * linvel.x + dir.y * linvel.y
            smoothing_fac = 40 / max(1, max_force)
            dotprod = np.clip(dotprod * smoothing_fac, -1, 1)
            self.body.ApplyForce(-dir * dotprod * max_force, pos, wake=True)

        front_tire_pos = self.get_front_tires_pos()
        front_tire_dir = self.get_front_tire_dir()
        apply_linear_friction(front_tire_pos, front_tire_dir, 0.1)
        apply_lateral_friction(front_tire_pos, front_tire_dir, 60)
        apply_brakes(front_tire_pos, front_tire_dir, self.brake * 60)

        back_tire_pos = self.get_back_tires_pos()
        back_tire_dir = self.get_car_direction()
        apply_linear_friction(back_tire_pos, back_tire_dir, 0.1)
        apply_lateral_friction(back_tire_pos, back_tire_dir, 60)
        apply_brakes(back_tire_pos, back_tire_dir, self.brake * 60)

        a = self.acceleration_coeff
        speed = self.body.linearVelocity.length
        acceleration = a * (self.theor_max_speed - speed) * self.body.mass
        if self.fwd and self.rwd:
            acceleration /= 2
        if self.fwd:
            self.body.ApplyForce(front_tire_dir * self.throttle * acceleration, front_tire_pos, wake=True)
        if self.rwd:
            self.body.ApplyForce(back_tire_dir * self.throttle * acceleration, back_tire_pos, wake=True)

    def get_car_direction(self):
        return Box2D.b2Vec2(np.cos(self.body.angle), np.sin(self.body.angle))

    def get_front_tires_pos(self):
        return self.body.worldCenter + self.get_car_direction() * 0.75

    def get_front_tire_dir(self):
        return Box2D.b2Vec2(np.cos(self.body.angle + self.steer_angle), np.sin(self.body.angle + self.steer_angle))

    def get_back_tires_pos(self):
        return self.body.worldCenter - self.get_car_direction() * 0.75

    def get_forward_speed(self):
        forward_dir = self.get_car_direction()
        vel = self.body.linearVelocity
        return vel.x * forward_dir.x + vel.y * forward_dir.y

    def get_vertices(self):
        vertices = self.body.fixtures[0].shape.vertices
        vertices = [(self.body.GetWorldPoint(Box2D.b2Vec2(x, y))) for x, y in vertices]
        return vertices

    def destroy(self):
        self.b2world.DestroyBody(self.body)
