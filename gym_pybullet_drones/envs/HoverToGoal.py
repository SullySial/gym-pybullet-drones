from collections import deque
import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from scipy.signal import butter, filtfilt

class HoverToGoal(BaseRLAviary):
    """Single agent RL problem: hover towards random xyz position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN, # <--- We could POSSIBLY change this observation into a RGB camera vision
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.TARGET_POS = np.array([1,1,1]) #### <--- Here's the line that indicates goal position (in this case, I changed x=1, y=2, z=3 for drone)
        self.TARGET_ORIENTATION = np.array([0,0,0])
        self.INIT_XYZS = np.array([0,0,0])
        self.EPISODE_LEN_SEC = 20            #### <--- Change the length of the episode in seconds 
        self.LOG_ANGULAR_VELOCITY = np.zeros((1, 3))
        self.LOG_RPMS = np.zeros((1, 4))
        self.ACTION_BUFFER_SIZE = int(ctrl_freq // 2)
        self.filtered_action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        
    # Adapted from piratax007's code for ActionsFilter.py
    ################################################################################
        
    def _target_error(self, state):
        return (np.linalg.norm(self.TARGET_POS - state[0:3]) +
                np.linalg.norm(self.TARGET_ORIENTATION - state[7:10]))
    
    ################################################################################

    def _is_away_from_exploration_area(self, state):
        return (np.linalg.norm(self.INIT_XYZS[0][0:2] - state[0:2]) >
                np.linalg.norm(self.INIT_XYZS[0][0:2] - self.TARGET_POS[0:2]) + 0.025 or
                state[2] > self.TARGET_POS[2] + 0.025)
    
    ################################################################################
    
    def _is_closed(self, state):
        return np.linalg.norm(state[0:3] - self.TARGET_POS[0:3]) < 0.025
        
    ################################################################################
    
    def _performance(self, state):
        if self._is_closed(state) and state[7]**2 + state[8]**2 < 0.01:
            return 2
        
        return -(state[7]**2 + state[8]**2)
    
    ################################################################################

    def _get_previous_current_we(self, current_state):
        if np.shape(self.LOG_ANGULAR_VELOCITY)[0] > 2:
            self.LOG_ANGULAR_VELOCITY = np.delete(self.LOG_ANGULAR_VELOCITY, 0, axis=0)

        return np.vstack((self.LOG_ANGULAR_VELOCITY, current_state[13:16]))

    ################################################################################

    def _get_we_differences(self, state):
        log = self._get_previous_current_we(state)
        differences = {
            'roll': log[0][0] - log[1][0],
            'pitch': log[0][1] - log[1][1],
            'yaw': log[0][2] - log[1][2],
        }
        return differences
    
    ################################################################################

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        # ret = max(0, 2 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)
        # return ret
        we_differences = self._get_we_differences(state)
        ret = (25 - 20 * self._target_error(state) -
               100 * (1 if self._is_away_from_exploration_area(state) else -0.25) +
               20 * self._performance(state) -
               18 * (we_differences['roll']**2 + we_differences['pitch']**2 + we_differences['yaw']**2))
        return ret
    
    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .03 and np.sqrt(state[7]**2 + state[8]**2) < 0.05: # < --- Could change to higher number
            return True
        else:
            return False
    
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        # if (abs(state[0]) > 5 or abs(state[1]) > 5 or state[2] > 5 # Truncate when the drone is too far away
        #      or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        # ):
        #     return True
        if (np.linalg.norm(self.INIT_XYZS[0][0:2] - state[0:2]) >
                np.linalg.norm(self.INIT_XYZS[0][0:2] - self.TARGET_POS[0:2]) + 1 or
                state[2] > self.TARGET_POS[2] + 1 or
                abs(state[7]) > .4 or abs(state[8]) > .4):
            return True

        
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
        
    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

    def _observationSpace(self):
        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([[lo, lo, 0, lo, lo, lo, lo, lo, lo, lo, lo, lo]])
        obs_upper_bound = np.array([[hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi]])
        act_lo = -1
        act_hi = +1
        for _ in range(self.ACTION_BUFFER_SIZE):
            obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo, act_lo, act_lo, act_lo]])])
            obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi, act_hi, act_hi, act_hi]])])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################

    def _computeObs(self):
        obs_12 = np.zeros((self.NUM_DRONES, 12))
        for i in range(self.NUM_DRONES):
            obs = self._getDroneStateVector(i)
            obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12, )
        ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
        for i in range(self.ACTION_BUFFER_SIZE):
            ret = np.hstack([ret, np.array([self.filtered_action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
        return ret

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """
        size = 4
        act_lower_bound = np.array([-1*np.ones(size) for _ in range(self.NUM_DRONES)])
        act_upper_bound = np.array([+1*np.ones(size) for _ in range(self.NUM_DRONES)])
        #
        for i in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES, size)))
            self.filtered_action_buffer.append(np.zeros((self.NUM_DRONES, size)))
        #
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    ################################################################################
    @staticmethod
    def butter_lowpass(cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    ################################################################################

    def apply_lowpass_filter(self, data, cutoff_frequency, sampling_frequency, filter_order=5):
        b, a = self.butter_lowpass(cutoff_frequency, sampling_frequency, order=filter_order)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    ################################################################################

    def apply_filter_to_actions(self, buffer, actions, cutoff_frequency, sampling_frequency, filter_order=5):
        flatted_buffer = np.concatenate(buffer).ravel().tolist()
        temp_buffer = np.concatenate((flatted_buffer[-56:], actions)).ravel().tolist()
        filtered_temp_buffer = self.apply_lowpass_filter(temp_buffer, cutoff_frequency, sampling_frequency, filter_order)
        filtered_actions = filtered_temp_buffer[-len(actions):]
        return np.array([filtered_actions])

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent
        RPMs, desired thrust and torques, the next target position to reach
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        self.action_buffer.append(action)
        # print(f"############ ACTIONS: {action} #############")
        action = self.apply_filter_to_actions(self.action_buffer, action[0], 14000, 30000)
#         print(f"######## FILTERED ACTIONS: {action} ###########")
        self.filtered_action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES, 4))
        for k in range(action.shape[0]):
            target = action[k, :]
            rpm[k, :] = np.array(self.HOVER_RPM * (1+0.05*target))
        return rpm

    ################################################################################