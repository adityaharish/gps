""" Hyperparameters for pendulum trajectory optimization experiment. """
from __future__ import division

import os.path
from datetime import datetime
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.prx.agent_prx import AgentPRX
# from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
#from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
#from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr
#from gps.gui.target_setup_gui import load_pose_from_npz
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, \
        TRIAL_ARM, AUXILIARY_ARM, JOINT_SPACE
from gps.utility.general_utils import get_ee_points
from gps.gui.config import generate_experiment_info

from gps.utility.make_angular_x0_set import MakeAngularX0Set

##
from caffe.proto.caffe_pb2 import SolverParameter, TRAIN, TEST
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM

# meaningless?
EE_POINTS = np.array([[]])

SENSOR_DIMS = {
    'FULL_STATE': 2,
    ACTION: 1,
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/pendulum_mdgps/'


common = {
    'experiment_name': 'pendulum_mdgps' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 50, # HOW MANY ENTRIES OF x0s TO ACTUALLY USE.
    'iterations': 10,
}
number_of_samples = 10

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])


####### INITIAL CONDITIONS ######
x0s   = [
        np.array([0.0, 0.0]),
        ]

# IC set using random angular pos & vel for each DOF.
maxtorque = 7
x0s = MakeAngularX0Set(common['conditions'],SENSOR_DIMS[ACTION],maxtorque) 
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
    'x0': x0s,
    'pos_body_idx': np.array([]),     #unneeded?
    'pos_body_offset': np.array([]),  #unneeded?
    'sensor_dims': SENSOR_DIMS,
    'state_include': ['FULL_STATE'],
    'obs_include': ['FULL_STATE'],
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

#algorithm = {
#    'type': AlgorithmTrajOpt,
#    'conditions': common['conditions'],
#}
algorithm = {
    'type': AlgorithmMDGPS,
    'conditions': common['conditions'],
    'iterations': common['iterations'],
    'kl_step': 1.0,
    'min_step_mult': 0.5,
    'max_step_mult': 3.0,
    'policy_sample_mode': 'replace',
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
    'init_var': 0.05, # CHANGED FROM 0.1.  variance of random control sampling?
    'stiffness': 0.01,
    'dt': agent['dt'],
    'T': agent['T'],
}


algorithm['dynamics'] = {
    'type': DynamicsLRPrior, # LR = linear regression
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

#algorithm['policy_opt'] = {}
algorithm['policy_opt'] = {
    'type': PolicyOptCaffe,
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
    'verbose_policy_trials': 0, # EXISTENCE IS USED AS A FLAG IN GPS_MAIN
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
