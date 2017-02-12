""" Hyperparameters for pendulum trajectory optimization experiment. """
from __future__ import division

import os.path
from datetime import datetime
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.prx.agent_prx import AgentPRX
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.gui.target_setup_gui import load_pose_from_npz
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, \
        TRIAL_ARM, AUXILIARY_ARM, JOINT_SPACE
from gps.utility.general_utils import get_ee_points
from gps.gui.config import generate_experiment_info

from gps.utility.make_angular_x0_set import MakeAngularX0Set

# Target states
EE_POINTS = np.array([[]])

SENSOR_DIMS = {
    'FULL_STATE': 2,
    ACTION: 1,
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/pendulum_test/'


common = {
    'experiment_name': 'pendulum_test' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1, # HOW MANY ENTRIES OF x0s TO ACTUALLY USE.
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])


####### INITIAL CONDITIONS ######
x0s   = [
        np.array([0.0, 0.0]),
        ]

#x0s = MakeAngularX0Set(common['conditions'],SENSOR_DIMS[ACTION],7) # random angular pos & vel for each DOF.

###### GOAL ######
xtarg = np.array([np.pi/2.0, 0.0])

agent = {
    'type': AgentPRX,
    'render' : False,
    'ee_points_tgt' : EE_POINTS,
    'dt': 0.1, # control dt
    'rk': 0,
    'conditions': common['conditions'],
    'reset_conditions': x0s,
    'T': 100, # Number of steps
    'x0': x0s[0],
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'sensor_dims': SENSOR_DIMS,
    'state_include': ['FULL_STATE'],
    'obs_include': [],
}

###### COST FUNCTIONS ######
action_cost = {
    'type': CostAction,
    'wu': np.array([1])
}

state_cost = {
    'type': CostState,
    'data_types' : {
        'FULL_STATE': {
            'wp': np.array([1, 1]),
            'target_state': xtarg,
        },
    },
}

algorithm = {
    'type': AlgorithmMDGPS,
    'conditions': common['conditions'],
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost],
    'weights': [1e-9, 1.0],
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains': np.zeros(SENSOR_DIMS[ACTION]),
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 10.0, # CHANGED FROM 0.1
    'stiffness': 0.01,
    'dt': agent['dt'],
    'T': agent['T'],
}


algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {}

config = {
    'iterations': 16,
    'num_samples': 50,
    'verbose_trials': 0,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
