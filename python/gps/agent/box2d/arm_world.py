""" This file defines an environment for the Box2D 2 Link Arm simulator. """
import Box2D as b2
import numpy as np
from framework import Framework
from gps.agent.box2d.settings import fwSettings
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS

class ArmWorld(Framework):
    """ This class defines the 2 Link Arm and its environment."""
    name = "2 Link Arm"
    def __init__(self, x0, target, render):
        self.render = render
        if self.render:
            super(ArmWorld, self).__init__()
        else:
            self.world = b2.b2World(gravity=(0, -10), doSleep=True)

        # self.world.gravity = (0.0, -9.18)
        self.world.gravity = (0.0, 0.0)

        fixture_length = 5
        self.fixture_length = 5
        self.x0 = x0
        
        square_fixture = b2.b2FixtureDef(
            shape=b2.b2PolygonShape(box=(1, 1)),
            density=100.0,
            #friction=1,
        )

        self.x_base = 0
        self.y_base = 15
        self.base = self.world.CreateBody(
            position=(self.x_base, self.y_base),
            fixtures=square_fixture,
        )

        blob_fixture = b2.b2FixtureDef(
            shape=b2.b2CircleShape(radius=1.5),
            density=1#/(np.pi*2.25),
        )

        self.body1 = self.world.CreateDynamicBody(
            fixtures = blob_fixture,
            position = (0, 0),
            #position = (self.x_base + self.fixture_length * np.sin(x0[0]),  self.y_base + -1 * self.fixture_length * np.cos(x0[0])),
            angle=x0[0],
        )
        
        _angle = target[0]
        self.target1 = self.world.CreateDynamicBody(
            fixtures=blob_fixture,
            position = (self.x_base + (self.fixture_length * np.sin(_angle)), self.y_base - (self.fixture_length * np.cos(_angle))),
            angle=_angle,
        )

        self.set_joint_angles(self.body1, x0[0], x0[1])
        self.set_joint_angles(self.target1, target[0], target[1])

        _angle = x0[0]
        self.joint1 = self.world.CreateRevoluteJoint(
            bodyA=self.base,
            bodyB=self.body1,
            localAnchorA=(self.x_base, self.y_base),
            localAnchorB=(self.x_base + (self.fixture_length * np.sin(_angle)), self.y_base - (self.fixture_length * np.cos(_angle))),
            enableMotor=True,
            maxMotorTorque=25,            
            enableLimit=True,
        )

        self.target1.active = False
        self.joint1.motorSpeed = 0#x0[1]

    def set_joint_angles(self, body, angle, speed):
        """ Converts the given absolute angle of the arms to joint angles"""
        #pos = self.base.GetWorldPoint((0, 0))
        body.angle = angle
        body.speed = speed
        # new_pos = body.GetWorldPoint((0, self.fixture_length))
        body.position = (self.x_base + (self.fixture_length * np.sin(angle)), self.y_base - (self.fixture_length * np.cos(angle)))

    def run(self):
        """Initiates the first time step
        """
        if self.render:
            super(ArmWorld, self).run()
        else:
            self.run_next(None)

    def run_next(self, action):
        """Moves forward in time one step. Calls the renderer if applicable."""
        if self.render:
            super(ArmWorld, self).run_next(action)
        else:
            if action is not None:
                self.joint1.motorSpeed = action[0]
                # self.joint2.motorSpeed = action[1]
            self.world.Step(1.0 / fwSettings.hz, fwSettings.velocityIterations,
                            fwSettings.positionIterations)

    def Step(self, settings, action):
        """Moves forward in time one step. Called by the renderer"""
        self.joint1.motorSpeed = action[0]
        # self.joint2.motorSpeed = action[1]

        super(ArmWorld, self).Step(settings)

    def reset_world(self):
        """Returns the world to its intial state"""
        self.world.ClearForces()
        self.joint1.motorSpeed = self.x0[1]
        # self.joint2.motorSpeed = 0
        self.body1.linearVelocity = (0, 0)
        self.body1.angularVelocity = 0
        # self.body2.linearVelocity = (0, 0)
        # self.body2.angularVelocity = 0
        self.set_joint_angles(self.body1, self.x0[0], self.x0[1])


    def get_state(self):
        """Retrieves the state of the point mass"""
        # state = {JOINT_ANGLES: np.array([self.joint1.angle,
        #                                  self.joint2.angle]),
        #          JOINT_VELOCITIES: np.array([self.joint1.speed,
        #                                      self.joint2.speed]),
        #          END_EFFECTOR_POINTS: np.append(np.array(self.body2.position),[0])}
        state = { 'FULL_STATE': np.array([self.joint1.angle, self.joint1.speed])}
        return state

