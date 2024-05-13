import os
from sys import platform
import time
import collections
from datetime import datetime
import xml.etree.ElementTree as etxml
import pkg_resources
from PIL import Image
# import pkgutil
# egl = pkgutil.get_loader('eglRenderer')
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType
from gym_pybullet_drones.assets.goal import Goal

class SimpleBase(gym.Env):
    """Base class for "drone DQN" Gym environments."""

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results'
                 ):
        """Initialization of a generic aviary environment.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
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
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        vision_attributes : bool, optional
            Whether to allocate the attributes needed by vision-based aviary subclasses.

        """
        #### Targets (Changeable) #############################################
        self.TARGET_POS = np.array([2,2,2]) #### <--- Here's the line that indicates goal position (in this case, I changed x=1, y=2, z=3 for drone)
        self.TARGET_ORIENTATION = np.array([0,0,0])
        self.EPISODE_LEN_SEC = 20            #### <--- Change the length of the episode in seconds 
        self.LOG_ANGULAR_VELOCITY = np.zeros((1, 3))
        self.LOG_RPMS = np.zeros((1, 4))
        self.goal_object = None
        self.goal = None
        #### Constants #############################################
        self.G = 9.8
        self.RAD2DEG = 180/np.pi
        self.DEG2RAD = np.pi/180
        self.CTRL_FREQ = ctrl_freq
        self.PYB_FREQ = pyb_freq
        if self.PYB_FREQ % self.CTRL_FREQ != 0:
            raise ValueError('[ERROR] in BaseAviary.__init__(), pyb_freq is not divisible by env_freq.')
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)
        self.CTRL_TIMESTEP = 1. / self.CTRL_FREQ
        self.PYB_TIMESTEP = 1. / self.PYB_FREQ
        #### Parameters ############################################
        self.NUM_DRONES = num_drones
        #### Options ###############################################
        self.DRONE_MODEL = drone_model
        self.GUI = gui
        self.RECORD = record
        self.PHYSICS = physics
        self.OBSTACLES = obstacles
        self.USER_DEBUG = user_debug_gui
        self.URDF = self.DRONE_MODEL.value + ".urdf"
        self.OUTPUT_FOLDER = output_folder
        #### Load the drone properties from the .urdf file #########
        self.M, \
        self.L, \
        self.THRUST2WEIGHT_RATIO, \
        self.J, \
        self.J_INV, \
        self.KF, \
        self.KM, \
        self.COLLISION_H,\
        self.COLLISION_R, \
        self.COLLISION_Z_OFFSET, \
        self.MAX_SPEED_KMH, \
        self.GND_EFF_COEFF, \
        self.PROP_RADIUS, \
        self.DRAG_COEFF, \
        self.DW_COEFF_1, \
        self.DW_COEFF_2, \
        self.DW_COEFF_3 = self._parseURDFParameters()
        print("[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:\n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f},\n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f},\n[INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f},\n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}".format(
            self.M, self.L, self.J[0,0], self.J[1,1], self.J[2,2], self.KF, self.KM, self.THRUST2WEIGHT_RATIO, self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS, self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1, self.DW_COEFF_2, self.DW_COEFF_3))
        #### Compute constants #####################################
        self.GRAVITY = self.G*self.M
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4*self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.KF))
        self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MAX_XY_TORQUE = (self.L*self.KF*self.MAX_RPM**2)
        elif self.DRONE_MODEL == DroneModel.RACE:
            self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)
        #### Create a buffer for the last .5 sec of actions ########
        self.ACTION_BUFFER_SIZE = int(ctrl_freq//2)
        #### Connect to PyBullet ###################################
        if self.GUI:
            #### With debug GUI ########################################
            self.CLIENT = p.connect(p.GUI) # p.connect(p.GUI, options="--opengl2")
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]:
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=3,
                                         cameraYaw=-30,
                                         cameraPitch=-30,
                                         cameraTargetPosition=[0, 0, 0],
                                         physicsClientId=self.CLIENT
                                         )
            ret = p.getDebugVisualizerCamera(physicsClientId=self.CLIENT)
            print("viewMatrix", ret[2])
            print("projectionMatrix", ret[3])
            
        else:
            #### Without debug GUI #####################################
            self.CLIENT = p.connect(p.DIRECT)
        #### Set initial poses #####################################
        if initial_xyzs is None:
            self.INIT_XYZS = np.vstack([np.array([x*4*self.L for x in range(self.NUM_DRONES)]), \
                                        np.array([y*4*self.L for y in range(self.NUM_DRONES)]), \
                                        np.ones(self.NUM_DRONES) * (self.COLLISION_H/2-self.COLLISION_Z_OFFSET+.1)]).transpose().reshape(self.NUM_DRONES, 3)
        elif np.array(initial_xyzs).shape == (self.NUM_DRONES,3):
            self.INIT_XYZS = initial_xyzs
        else:
            print("[ERROR] invalid initial_xyzs in BaseAviary.__init__(), try initial_xyzs.reshape(NUM_DRONES,3)")
        if initial_rpys is None:
            self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        elif np.array(initial_rpys).shape == (self.NUM_DRONES, 3):
            self.INIT_RPYS = initial_rpys
        else:
            print("[ERROR] invalid initial_rpys in BaseAviary.__init__(), try initial_rpys.reshape(NUM_DRONES,3)")
        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()    

    ################################################################################
    
    def reset(self,
              seed : int = None,
              options : dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """

        # TODO : initialize random number generator with seed

        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Return the initial observation ########################
        # initial_obs = self._computeObs()
        initial_obs = self._computeExtendedObs()
        initial_info = self._computeInfo()
        return np.array(initial_obs) #, dtype=np.float32) # initial_obs, initial_info
    
    ################################################################################
    
    def step(self,
             action
             ):
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is over, check the specific implementation of `_computeTerminated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is truncated, check the specific implementation of `_computeTruncated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is trunacted, always false.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        fwd = [0,0,0,0,1,-1,0]
        left = [0,0,1,-1,0,0,0]
        up = [1,-1,0,0,0,0,0]
        x = fwd[action]
        y = left[action]
        z = up[action]
        action = [x, y, z]
        self.apply_action(action)
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Prepare the return values #############################
        # obs = self._computeObs()
        obs = self._computeExtendedObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        # truncated = self._computeTruncated()
        info = self._computeInfo()
        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)
        return obs, reward, terminated, dict()


    ################################################################################
        
    def apply_action(self, action):
        # clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
        clipped_action = self._preprocessAction(action)
        #### Repeat for as many as the aggregate physics steps #####
        for _ in range(self.PYB_STEPS_PER_CTRL):
            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()
            #### Step the simulation using the desired physics update ##
            for i in range (self.NUM_DRONES):
                if self.PHYSICS == Physics.PYB:
                    # self._physics(clipped_action[i, :], i)
                    self._physics(clipped_action, i)
            #### PyBullet computes the new state ###
            p.stepSimulation(physicsClientId=self.CLIENT)
            #### Buffer with brake action ####
            # action_buffer = [0,0,0]
            # clipped_action_buffer = self._preprocessAction(action_buffer)
            # self._physics(clipped_action_buffer,0)
            # p.stepSimulation(physicsClientId=self.CLIENT)

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
        rpm = np.zeros(4)
        value = 0
        if action[0] !=0:
            value = action[0]
            if value == 1:
                # Move to the left
                rpm[0] = self.HOVER_RPM * (1+0.0125*abs(value))
                rpm[1] = self.HOVER_RPM 
                rpm[2] = self.HOVER_RPM * (1+0.0125*abs(value))
                rpm[3] = self.HOVER_RPM 
                # print("[INFO] Chosen RPM (left): ", rpm)

                return rpm
            else:
                # Move to the right
                rpm[0] = self.HOVER_RPM 
                rpm[1] = self.HOVER_RPM * (1+0.0125*abs(value))
                rpm[2] = self.HOVER_RPM 
                rpm[3] = self.HOVER_RPM * (1+0.0125*abs(value))
                # print("[INFO] Chosen RPM (right): ", rpm)

                return rpm
        elif action[1] !=0:
            value = action[1]
            if value == 1:
                # Move forward
                rpm[0] = self.HOVER_RPM * (1+0.0125*abs(value))
                rpm[1] = self.HOVER_RPM * (1+0.0125*abs(value))
                rpm[2] = self.HOVER_RPM 
                rpm[3] = self.HOVER_RPM 
                # print("[INFO] Chosen RPM (forward): ", rpm)

                return rpm
            else:
                # Move backwards
                rpm[0] = self.HOVER_RPM 
                rpm[1] = self.HOVER_RPM 
                rpm[2] = self.HOVER_RPM * (1+0.0125*abs(value))
                rpm[3] = self.HOVER_RPM * (1+0.0125*abs(value))
                # print("[INFO] Chosen RPM (backward): ", rpm)

                return rpm
        elif action[2] !=0:
            value = action[2]
            if value == 1:
                # Move up
                rpm[0] = self.HOVER_RPM * (1+0.0125*abs(value))
                rpm[1] = self.HOVER_RPM * (1+0.0125*abs(value))
                rpm[2] = self.HOVER_RPM * (1+0.0125*abs(value))
                rpm[3] = self.HOVER_RPM * (1+0.0125*abs(value))
                # print("[INFO] Chosen RPM (up): ", rpm)

                return rpm
            else:
                # Move down
                rpm[0] = self.HOVER_RPM * (1+0.0125*value)
                rpm[1] = self.HOVER_RPM * (1+0.0125*value)
                rpm[2] = self.HOVER_RPM * (1+0.0125*value)
                rpm[3] = self.HOVER_RPM * (1+0.0125*value)
                # print("[INFO] Chosen RPM (down): ", rpm)

                return rpm
        elif action == [0,0,0]:
            rpm[0] = self.HOVER_RPM 
            rpm[1] = self.HOVER_RPM 
            rpm[2] = self.HOVER_RPM 
            rpm[3] = self.HOVER_RPM 
            # print("[INFO] Chosen RPM (brake): ", rpm)

            return rpm
        else:
            print("[WARNING] no RPM chosen")
            return False
            
    ################################################################################
    
    def _physics(self,
                 rpm,
                 nth_drone
                 ):
        """Base PyBullet physics implementation.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        forces = np.array(rpm**2)*self.KF
        # print("forces: ", forces)
        torques = np.array(rpm**2)*self.KM
        # print("torques: ", torques)
        if self.DRONE_MODEL == DroneModel.RACE:
            torques = -torques
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):
            p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                 i,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.CLIENT
                                 )
        # p.applyExternalTorque(self.DRONE_IDS[nth_drone],
        #                       4,
        #                       torqueObj=[0, 0, z_torque],
        #                       flags=p.LINK_FRAME,
        #                       physicsClientId=self.CLIENT
        #                       )

    ################################################################################

    def _updateAndStoreKinematicInformation(self):
        """Updates and stores the drones kinemaatic information.

        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).

        """
        for i in range (self.NUM_DRONES):
            self.pos[i], self.quat[i] = p.getBasePositionAndOrientation(self.DRONE_IDS[i], physicsClientId=self.CLIENT)
            self.rpy[i] = p.getEulerFromQuaternion(self.quat[i])
            self.vel[i], self.ang_v[i] = p.getBaseVelocity(self.DRONE_IDS[i], physicsClientId=self.CLIENT)
    
    ################################################################################

    def _getDroneStateVector(self,
                             nth_drone
                             ):
        """Returns the state vector of the n-th drone.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        Returns
        -------
        ndarray 
            (20,)-shaped array of floats containing the state vector of the n-th drone.
            Check the only line in this method and `_updateAndStoreKinematicInformation()`
            to understand its format.

        """
        state = np.hstack([self.pos[nth_drone, :], self.quat[nth_drone, :], self.rpy[nth_drone, :],
                           self.vel[nth_drone, :], self.ang_v[nth_drone, :], self.last_clipped_action[nth_drone, :]])
        return state.reshape(20,)

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """

        ############################################################
        #### OBS SPACE OF SIZE 12
        obs_12 = np.zeros((self.NUM_DRONES,12))
        for i in range(self.NUM_DRONES):
            #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
            obs = self._getDroneStateVector(i)
            obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
        ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
        #### Add action buffer to observation #######################
        for i in range(self.ACTION_BUFFER_SIZE):
            ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
        return ret
    
    ################################################################################

    def _computeExtendedObs(self):
        state = self._getDroneStateVector(0)
        goalpos, goalorn = p.getBasePositionAndOrientation(self.goal_object.goal)
        # dronePos = self.pos[:]
        # droneOrn = self.quat[:]
        dronePos = state[0:3]
        droneOrn = state[3:7]
        # print("dronepos: ", dronePos, ", droneorn: ", droneOrn)
        invDronePos, invDroneOrn = p.invertTransform(dronePos, droneOrn)
        goalPosInDrone, goalOrnInDrone = p.multiplyTransforms(invDronePos, invDroneOrn, goalpos, goalorn)
        # goal_pos_relative = [self.TARGET_POS[i] - state[i] for i in range(3)]
        # observation = [goal_pos_relative[0], goal_pos_relative[1], goal_pos_relative[2]]
        observation = [goalPosInDrone[0], goalPosInDrone[1], goalPosInDrone[2]]
        return observation
        
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

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        # state = self._getDroneStateVector(0)
        # # ret = max(0, 2 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)
        # # return ret
        # ret = -np.linalg.norm(self.TARGET_POS-state[0:3])
        # if np.linalg.norm(self.TARGET_POS-state[0:3]) < 0.9:
        #     ret+= 50
        #     print("[INFO] Reached Target")

        state = self._getDroneStateVector(0)
        target_distance = np.linalg.norm(self.TARGET_POS - state[0:3])
        
        # Calculate tilt angle (assuming roll and pitch angles are stored in state[6:8])
        roll_angle = state[7]
        pitch_angle = state[8]
        tilt_angle = np.sqrt(roll_angle ** 2 + pitch_angle ** 2)

        max_allowed_tilt_angle = np.radians(45)
        tilt_penalty_factor = 10
        # Penalize for excessive tilt angle
        tilt_penalty = max(0, tilt_angle - max_allowed_tilt_angle) * tilt_penalty_factor
        
        # Reward based on distance to target
        ret = -target_distance
        
        # Add a bonus when target is reached
        if target_distance < 0.9:
            ret += 50
            print("[INFO] Reached Target")

        # Add the tilt penalty to the reward
        ret -= tilt_penalty

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
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < 0.9: # < --- Could change to higher number
            return True
        else:
            truncated = self._computeTruncated()
            return truncated
        
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
                np.linalg.norm(self.INIT_XYZS[0][0:2] - self.TARGET_POS[0:2]) + 3 or
                state[2] > self.TARGET_POS[2] + 2 or
                abs(state[7]) > 1.2 or abs(state[8]) > 1.2):
            return True

        
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
        
    ################################################################################

    def getDroneIds(self):
        """Return the Drone Ids.

        Returns
        -------
        ndarray:
            (NUM_DRONES,)-shaped array of ints containing the drones' ids.

        """
        return self.DRONE_IDS
    
    ################################################################################

    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.

        """
        #### Initialize/reset counters and zero-valued variables ###
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True
        self.X_AX = -1*np.ones(self.NUM_DRONES)
        self.Y_AX = -1*np.ones(self.NUM_DRONES)
        self.Z_AX = -1*np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1*np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM=False
        self.last_input_switch = 0
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))
        self.gui_input = np.zeros(4)
        #### Initialize the drones kinemaatic information ##########
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((self.NUM_DRONES, 3))
        #### Set PyBullet's parameters #############################
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        #### Load ground plane, drone and obstacles models #########
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)

        self.DRONE_IDS = np.array([p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF),
                                              self.INIT_XYZS[i,:],
                                              p.getQuaternionFromEuler(self.INIT_RPYS[i,:]),
                                              flags = p.URDF_USE_INERTIA_FROM_FILE,
                                              physicsClientId=self.CLIENT
                                              ) for i in range(self.NUM_DRONES)])
        # Set goal to target coordinates
        self.goal = (self.TARGET_POS[0], self.TARGET_POS[1], self.TARGET_POS[2])
        # Visual element of the goal
        self.goal_object = Goal(p, self.goal)
        #### Remove default damping #################################
        # for i in range(self.NUM_DRONES):
        #     p.changeDynamics(self.DRONE_IDS[i], -1, linearDamping=0, angularDamping=0)
        #### Show the frame of reference of the drone, note that ###
        #### It severly slows down the GUI #########################
        if self.GUI and self.USER_DEBUG:
            for i in range(self.NUM_DRONES):
                self._showDroneLocalAxes(i)
        #### Disable collisions between drones' and the ground plane
        #### E.g., to start a drone at [0,0,0] #####################
        # for i in range(self.NUM_DRONES):
            # p.setCollisionFilterPair(bodyUniqueIdA=self.PLANE_ID, bodyUniqueIdB=self.DRONE_IDS[i], linkIndexA=-1, linkIndexB=-1, enableCollision=0, physicsClientId=self.CLIENT)
        if self.OBSTACLES:
            self._addObstacles()
    
    ################################################################################

    def _showDroneLocalAxes(self,
                            nth_drone
                            ):
        """Draws the local frame of the n-th drone in PyBullet's GUI.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        if self.GUI:
            AXIS_LENGTH = 2*self.L
            self.X_AX[nth_drone] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[AXIS_LENGTH, 0, 0],
                                                      lineColorRGB=[1, 0, 0],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.X_AX[nth_drone]),
                                                      physicsClientId=self.CLIENT
                                                      )
            self.Y_AX[nth_drone] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[0, AXIS_LENGTH, 0],
                                                      lineColorRGB=[0, 1, 0],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.Y_AX[nth_drone]),
                                                      physicsClientId=self.CLIENT
                                                      )
            self.Z_AX[nth_drone] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[0, 0, AXIS_LENGTH],
                                                      lineColorRGB=[0, 0, 1],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.Z_AX[nth_drone]),
                                                      physicsClientId=self.CLIENT
                                                      )
    
    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.

        """
        p.loadURDF("samurai.urdf",
                   physicsClientId=self.CLIENT
                   )
        p.loadURDF("duck_vhacd.urdf",
                   [-.5, -.5, .05],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        p.loadURDF("cube_no_rotation.urdf",
                   [-.5, -2.5, .5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        p.loadURDF("sphere2.urdf",
                   [0, 2, .5],
                   p.getQuaternionFromEuler([0,0,0]),
                   physicsClientId=self.CLIENT
                   )
    
    ################################################################################
    
    def _parseURDFParameters(self):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF)).getroot()
        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
               GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3
    
    ################################################################################
    
    def close(self):
        """Terminates the environment.
        """
        p.disconnect(physicsClientId=self.CLIENT)

    ################################################################################

    def getPyBulletClient(self):
        """Returns the PyBullet Client Id.

        Returns
        -------
        int:
            The PyBullet Client Id.

        """
        return self.CLIENT
    
    ################################################################################

    def _observationSpace(self):
        # lo = -np.inf
        # hi = np.inf
        # obs_lower_bound = np.array([[lo, lo, 0, lo, lo, lo, lo, lo, lo, lo, lo, lo]])
        # obs_upper_bound = np.array([[hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi]])
        # act_lo = -1
        # act_hi = +1
        # for _ in range(self.ACTION_BUFFER_SIZE):
        #     obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo, act_lo, act_lo, act_lo]])])
        #     obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi, act_hi, act_hi, act_hi]])])
        observation_space = gym.spaces.box.Box(
            low=np.array([-40, -40, -40], dtype=np.float32),
            high=np.array([40, 40, 40], dtype=np.float32))
        
        return observation_space
    
    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Discrete
            A 1D array of size 7

        """
        self.action_space = gym.spaces.Discrete(7)

        return self.action_space 

    ################################################################################

    def render(self, mode='human'):
        if mode == "fp_camera":
            # Base information
            car_id = self.car.get_ids()
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                       nearVal=0.01, farVal=100)
            pos, ori = [list(l) for l in
                        self._p.getBasePositionAndOrientation(car_id)]
            pos[2] = 0.2

            # Rotate camera direction
            rot_mat = np.array(self._p.getMatrixFromQuaternion(ori)).reshape(3, 3)
            camera_vec = np.matmul(rot_mat, [1, 0, 0])
            up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
            view_matrix = self._p.computeViewMatrix(pos, pos + camera_vec, up_vec)

            # Display image
            # frame = self._p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
            # frame = np.reshape(frame, (100, 100, 4))
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                      height=RENDER_HEIGHT,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame
            # self.rendered_img.set_data(frame)
            # plt.draw()
            # plt.pause(.00001)

        elif mode == "tp_camera":
            car_id = self.car.get_ids()
            base_pos, orn = self._p.getBasePositionAndOrientation(car_id)
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                    distance=20.0,
                                                                    yaw=40.0,
                                                                    pitch=-35,
                                                                    roll=0,
                                                                    upAxisIndex=2)
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                             aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                             nearVal=0.1,
                                                             farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                      height=RENDER_HEIGHT,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame
        else:
            return np.array([])