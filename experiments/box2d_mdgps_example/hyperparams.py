""" Hyperparameters for Box2d Point Mass."""
from __future__ import division

import os.path
from datetime import datetime
import numpy as np
import math
from gps import __file__ as gps_filepath
from gps.agent.box2d.agent_box2d import AgentBox2D
from gps.agent.box2d.arm_world import ArmWorld
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.gui.config import generate_experiment_info
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, ACTION
from gps.utility.make_angular_x0_set import MakeAngularX0Set



SENSOR_DIMS = {
    'FULL_STATE': 2,
    ACTION: 1,
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/box2d_mdgps_example/'


common = {
    'experiment_name': 'box2d_mdgps_example' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 50,
    'iterations': 10,
    'conditions': 5,
    'iterations': 3,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

maxvel = 7

number_of_samples = 10

number_of_samples = 3
###### INITIAL STATES ######
x0s = MakeAngularX0Set(common['conditions'],SENSOR_DIMS[ACTION],maxvel)

x0s = [np.array([0.0, 0.0])] * common['conditions']

###### GOAL ######
xtarg = np.array([np.pi/6.0, 0.0])

agent = {
    'type': AgentBox2D,
    'target_state' : xtarg,
    'world' : ArmWorld,
    'x0': x0s,
    'rk': 0,
    'dt': 0.05,
    'substeps': 1,
    'render': True,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'reset_conditions': x0s,
    'T': 20,
    'sensor_dims': SENSOR_DIMS,
    'state_include': ['FULL_STATE'],
    'obs_include': ['FULL_STATE'],
}

algorithm = {
    'type': AlgorithmMDGPS,
    'conditions': common['conditions'],
    'iterations': common['iterations'],
    'fixed_lg_step': 3,
    'kl_step': 5.0,
    'min_step_mult': 0.05,
    'max_step_mult': 3.0,
    'policy_sample_mode': 'replace',
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains': np.zeros(SENSOR_DIMS[ACTION]),
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 0.05,
    'stiffness': 0.01,
    'dt': agent['dt'],
    'T': agent['T'],
}

action_cost = {
    'type': CostAction,
    'wu': np.array([1])
}

state_cost = {
    'type': CostState,
    'data_types' : {
        'FULL_STATE': {
            'wp': np.array([1, 1]),
            'target_state': agent["target_state"],
        },
    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost],
    'weights': [1e-5, 1.0],
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

algorithm['policy_opt'] = {
    'type': PolicyOptCaffe,
    'iterations': 4000,
    'weights_file_prefix': EXP_DIR + 'policy',
    # DEFAULTS:             3,None,27,7,25,TRAIN)
    'network_arch_params': {
        'n_layers':2, # last layer has dim dim_output (appended)
        'dim_hidden':[20], # one entry fewer than n_layers
        'dim_input':SENSOR_DIMS['FULL_STATE'],
        'dim_output':SENSOR_DIMS[ACTION],
        'batch_size':25,
    },  
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': number_of_samples,
    'verbose_trials': 0,
    'verbose_policy_trials': 0,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
