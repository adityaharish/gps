""" This file defines an agent for the PRX environment. """
import copy
import time
import numpy as np

import rospy

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_PRX
from gps.agent.ros.ros_utils import ServiceEmulator, msg_to_sample, \
        policy_to_msg, tf_policy_to_action_msg, tf_obs_msg_to_numpy
# from gps.proto.gps_pb2 import FULL_STATE
from gps.sample.sample import Sample
from gps_agent_pkg.msg import  DataRequest
from prx_simulation.msg import \
         gps_state_command_msg, \
         gps_trial_command_msg, \
         gps_sample_result_msg

from gps.proto.gps_pb2 import ACTION

try:
    from gps.algorithm.policy.tf_policy import TfPolicy
except ImportError:  # user does not have tf installed.
    TfPolicy = None



class AgentPRX(Agent):
    """
    All communication between the algorithms and PRX is done through
    this class.
    """
    def __init__(self, hyperparams, init_node=True):
        """
        Initialize agent.
        Args:
            hyperparams: Dictionary of hyperparameters.
            init_node: Whether or not to initialize a new ROS node.
        """
        config = copy.deepcopy(AGENT_PRX)
        config.update(hyperparams)
        Agent.__init__(self, config)
        if init_node:
            rospy.init_node('gps_agent_prx_node')
        self._init_pubs_and_subs()
        self._seq_id = 0  # Used for setting seq in PRX commands.

        conditions = self._hyperparams['conditions']

        self.x0 = []
        for field in ('x0', 'ee_points_tgt', 'reset_conditions'):
            self._hyperparams[field] = setup(self._hyperparams[field],
                                             conditions)
        self.x0 = self._hyperparams['x0']

        r = rospy.Rate(1)
        r.sleep()

        self.use_tf = False
        self.observations_stale = True

    def _init_pubs_and_subs(self):
        self._trial_service = ServiceEmulator(
            self._hyperparams['trial_command_topic'], gps_trial_command_msg,
            self._hyperparams['sample_result_topic'], gps_sample_result_msg
        )
        self._reset_service = ServiceEmulator(
            self._hyperparams['reset_command_topic'], gps_state_command_msg,
            self._hyperparams['sample_result_topic'], gps_sample_result_msg
        )
        self._data_service = ServiceEmulator(
            self._hyperparams['data_request_topic'], DataRequest,
            self._hyperparams['sample_result_topic'], gps_sample_result_msg
        )

    def _get_next_seq_id(self):
        self._seq_id = (self._seq_id + 1) % (2 ** 32)
        return self._seq_id

    def get_data(self):
        """
        Request for the most recent value for data/sensor readings.
        Returns entire sample report (all available data) in sample.
        """
        request = DataRequest()
        request.id = self._get_next_seq_id()
        request.arm = 0 # NEED TO CHANGE MESSAGE STRUCTURE
        request.stamp = rospy.get_rostime()
        result_msg = self._data_service.publish_and_wait(request)
        # TODO - Make IDs match, assert that they match elsewhere here.
        sample = msg_to_sample(result_msg, self)
        return sample


    def reset(self, condition):
        """
        Reset the agent for a particular experiment condition.
        Args:
            condition: An index into hyperparams['reset_conditions'].
        """

        reset_command = gps_state_command_msg()                   
        reset_command.state = self._hyperparams['reset_conditions'][condition]
        reset_command.id = self._get_next_seq_id()
        timeout = self._hyperparams['trial_timeout']
        self._reset_service.publish_and_wait(reset_command, timeout=timeout)

#        time.sleep(2.0)  # useful for the real robot, so it stops completely

    def _get_new_action(self, policy, obs):
        return policy.act(None, obs, None, None)

    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """
        Reset and execute a policy and collect a sample.
        Args:
            policy: A Policy object.
            condition: Which condition setup to run.
            verbose: Unused for this agent.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
        Returns:
            sample: A Sample object.
        """
        if TfPolicy is not None:  # user has tf installed.
            if isinstance(policy, TfPolicy):
                self._init_tf(policy.dU)

        self.reset(condition)

        # Generate noise.
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        # Execute trial.
        trial_command = gps_trial_command_msg()
        trial_command.id = self._get_next_seq_id()
        trial_command.controller = policy_to_msg(policy, noise)
        trial_command.T = self.T
        trial_command.id = self._get_next_seq_id() # Why again?
        trial_command.frequency = self._hyperparams['frequency'] # 1/dt
#        trial_command.state_datatypes = self._hyperparams['state_include']
#        trial_command.obs_datatypes = self._hyperparams['state_include']

        if self.use_tf:
            raise AttributeError("agent_prx currently does not support tensorflow")

#        print "Publishing trial command."
        sample_msg = self._trial_service.publish_and_wait(
            trial_command, timeout=self._hyperparams['trial_timeout']
        )

#        print "Received sample msg # {}\n".format(sample_msg.id)
        # partly bypassing protobuf and datatype stuff
        # this could be done more elegantly.  maybe make sample_prx.

        sample = Sample(self)
        stateshape = np.array([self.T,self.dX])
        controlshape = np.array([self.T,self.dU])
        statedata = np.array(sample_msg.states).reshape(stateshape)
        controldata = np.array(sample_msg.controls).reshape(controlshape)
        sample.set('FULL_STATE',statedata)
        sample.set(ACTION,controldata)

        if save:
            self._samples[condition].append(sample)
        return sample

