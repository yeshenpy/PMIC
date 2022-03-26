import itertools
import numpy as np


class Node():
    def __init__(self, label, qpos_ids, qvel_ids, act_ids):
        self.label = label
        self.qpos_ids = qpos_ids
        self.qvel_ids = qvel_ids
        self.act_ids = act_ids
        pass

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.label


class HyperEdge():
    def __init__(self, *edges):
        self.edges = set(edges)

    def __contains__(self, item):
        return item in self.edges

    def __str__(self):
        return "HyperEdge({})".format(self.edges)

    def __repr__(self):
        return "HyperEdge({})".format(self.edges)


def get_joints_at_kdist(agent_id, agent_partitions, hyperedges, k=0, kagents=False,):
    """ Identify all joints at distance <= k from agent agent_id

    :param agent_id: id of agent to be considered
    :param agent_partitions: list of joint tuples in order of agentids
    :param edges: list of tuples (joint1, joint2)
    :param k: kth degree
    :param kagents: True (observe all joints of an agent if a single one is) or False (individual joint granularity)
    :return:
        dict with k as key, and list of joints at that distance
    """
    assert not kagents, "kagents not implemented!"

    agent_joints = agent_partitions[agent_id]

    def _adjacent(lst, kagents=False):
        # return all sets adjacent to any element in lst
        ret = set([])
        for l in lst:
            ret = ret.union(set(itertools.chain(*[e.edges.difference({l}) for e in hyperedges if l in e])))
        return ret

    seen = set([])
    new = set([])
    k_dict = {}
    for _k in range(k+1):
        if not _k:
            new = set(agent_joints)
        else:
            print(hyperedges)
            new = _adjacent(new) - seen
        seen = seen.union(new)
        k_dict[_k] = sorted(list(new), key=lambda x:x.label)
    return k_dict


def build_obs(k_dict, qpos, qvel, vec_len=None, add_global_pos=False):
    """Given a k_dict from get_joints_at_kdist, extract observation vector.

    :param k_dict: k_dict
    :param qpos: qpos numpy array
    :param qvel: qvel numpy array
    :param vec_len: if None no padding, else zero-pad to vec_len
    :return:
    observation vector
    """
    obs_qpos_lst = []
    obs_qvel_lst = []
    # for k in sorted(list(k_dict.keys())):
    #     for _t in k_dict[k]:
    #         qpos_ids = _t.qpos_ids
    #         qvel_ids = _t.qvel_ids
    #         if add_global_pos:
    #             qpos_ids = [0, 1, 2] + [qpos_ids] if not isinstance(qpos_ids, list) else qpos_ids
    #             qvel_ids = [0, 1, 2] + [qvel_ids] if not isinstance(qvel_ids, list) else qvel_ids
    #         obs_qpos_lst.append(qpos[qpos_ids])
    #         obs_qvel_lst.append(qvel[qvel_ids])

    if add_global_pos:
        for qpos_ids in [0, 1, 2]:
            obs_qpos_lst.append(qpos[qpos_ids])
        for qvel_ids in [0, 1, 2]:
            obs_qvel_lst.append(qvel[qvel_ids])
    for k in sorted(list(k_dict.keys())):
        for _t in k_dict[k]:
            if _t.qpos_ids is not None: # if node is observable
                obs_qpos_lst.append(qpos[_t.qpos_ids])
            if _t.qvel_ids is not None: # if node is observable
                obs_qvel_lst.append(qvel[_t.qvel_ids])

    ret = np.concatenate([obs_qpos_lst,
                          obs_qvel_lst])
    if vec_len is not None:
        pad = np.array((vec_len - len(obs_qpos_lst) - len(obs_qvel_lst))*[0])
        return np.concatenate([ret, pad])
    return ret


def build_actions(agent_partitions, k_dict):
    # Composes agent actions output from networks
    # into coherent joint action vector to be sent to the env.


    pass

def get_parts_and_edges(label, partitioning):
    if label in ["half_cheetah", "HalfCheetah-v2"]:

        # define Mujoco graph
        bthigh = Node("bthigh", 3, 3, 0)
        bshin = Node("bshin", 4, 4, 1)
        bfoot = Node("bfoot", 5, 5, 2)
        fthigh = Node("fthigh", 6, 6, 3)
        fshin = Node("fshin", 7, 7, 4)
        ffoot = Node("ffoot", 8, 8, 5)

        edges = [HyperEdge(bfoot, bshin),
                 HyperEdge(bshin, bthigh),
                 HyperEdge(bthigh, fthigh),
                 HyperEdge(fthigh, fshin),
                 HyperEdge(fshin, ffoot)]

        if partitioning == "2x3":
            parts = [(bfoot, bshin, bthigh),
                     (ffoot, fshin, fthigh)]
        elif partitioning == "6x1":
            parts = [(bfoot,), (bshin,), (bthigh,), (ffoot,), (fshin,), (fthigh,)]
        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges

    elif label in ["Ant-v2"]:

        # define Mujoco graph
        hip4 = Node("hip4", 3, 3, 0)
        ankle4 = Node("ankle4", 4, 4, 1)
        hip1 = Node("hip1", 5, 5, 2)
        ankle1 = Node("ankle1", 6, 6, 3)
        hip2 = Node("hip2", 7, 7, 4)
        ankle2 = Node("ankle2", 8, 8, 5)
        hip3 = Node("hip3", 9, 9, 6)
        ankle3 = Node("ankle3", 10, 10, 7)
        body = Node("body", None, None, None)

        edges = [HyperEdge(ankle4, hip4),
                 HyperEdge(ankle1, hip1),
                 HyperEdge(ankle2, hip2),
                 HyperEdge(ankle3, hip3),
                 HyperEdge(hip4, body),
                 HyperEdge(hip1, body),
                 HyperEdge(hip2, body),
                 HyperEdge(hip3, body),
                 ]

        if partitioning == "2x4": # neighbouring legs together
            parts = [(hip1, ankle1, hip2, ankle2),
                     (hip3, ankle3, hip4, ankle4)]
        elif partitioning == "2x4d": # diagonal legs together
            parts = [(hip1, ankle1, hip3, ankle3),
                     (hip2, ankle2, hip4, ankle4)]
        elif partitioning == "4x2":
            parts = [(hip1, ankle1),
                     (hip2, ankle2),
                     (hip3, ankle3),
                     (hip4, ankle4)]
        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges

    elif label in ["Hopper-v2"]:

        # define Mujoco-Graph
        thigh_joint = Node("thigh_joint", 3, 3, 0)
        leg_joint = Node("leg_joint", 4, 4, 1)
        foot_joint = Node("foot_joint", 5, 5, 2)

        edges = [HyperEdge(foot_joint, leg_joint),
                 HyperEdge(leg_joint, thigh_joint)]

        if partitioning == "3x1":
            parts = [(thigh_joint,),
                     (leg_joint,),
                     (foot_joint,)]

        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges

    elif label in ["Humanoid-v2", "HumanoidStandup-v2"]:

        # define Mujoco-Graph
        abdomen_y = Node("abdomen_y", 3, 3, 1) # act ordering bug in env -- double check!
        abdomen_z = Node("abdomen_z", 4, 4, 0)
        abdomen_x = Node("abdomen_x", 5, 5, 2)
        right_hip_x = Node("right_hip_x", 6, 6, 3)
        right_hip_z = Node("right_hip_z", 7, 7, 4)
        right_hip_y = Node("right_hip_y", 8, 8, 5)
        right_knee = Node("right_knee", 9, 9, 6)
        left_hip_x = Node("left_hip_x", 10, 10, 7)
        left_hip_z = Node("left_hip_z", 11, 11, 8)
        left_hip_y = Node("left_hip_y", 12, 12, 9)
        left_knee = Node("left_knee", 13, 13, 10)
        right_shoulder1 = Node("right_shoulder1", 14, 14, 11)
        right_shoulder2 = Node("right_shoulder2", 15, 15, 12)
        right_elbow = Node("right_elbow", 16, 16, 13)
        left_shoulder1 = Node("left_shoulder1", 17, 17, 14)
        left_shoulder2 = Node("left_shoulder2", 18, 18, 15)
        left_elbow = Node("left_elbow", 19, 19, 16)

        edges = [HyperEdge(abdomen_x, abdomen_y, abdomen_z),
                 HyperEdge(right_hip_x, right_hip_y, right_hip_z),
                 HyperEdge(left_hip_x, left_hip_y, left_hip_z),
                 HyperEdge(left_elbow, left_shoulder1, left_shoulder2),
                 HyperEdge(right_elbow, right_shoulder1, right_shoulder2),
                 HyperEdge(left_knee, left_hip_x, left_hip_y, left_hip_z),
                 HyperEdge(right_knee, right_hip_x, right_hip_y, right_hip_z),
                 HyperEdge(left_shoulder1, left_shoulder2, abdomen_x, abdomen_y, abdomen_z),
                 HyperEdge(right_shoulder1, right_shoulder2, abdomen_x, abdomen_y, abdomen_z),
                 HyperEdge(abdomen_x, abdomen_y, abdomen_z, left_hip_x, left_hip_y, left_hip_z),
                 HyperEdge(abdomen_x, abdomen_y, abdomen_z, right_hip_x, right_hip_y, right_hip_z),
                 ]

        if partitioning == "2x8": # 17 in total, so one action is a dummy (to be handled by pymarl)
            # isolate upper and lower body
            parts = [(left_shoulder1, left_shoulder2, abdomen_x, abdomen_y, abdomen_z,
                      right_shoulder1, right_shoulder2,
                      right_elbow, left_elbow),
                     (left_hip_x, left_hip_y, left_hip_z,
                      right_hip_x, right_hip_y, right_hip_z,
                      right_knee, left_knee)]
            # TODO: There could be tons of decompositions here

        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges

    elif label in ["Reacher-v2"]:

        # define Mujoco-Graph
        joint0 = Node("joint0", 0, 0, 0) # TODO: double-check ids
        joint1 = Node("joint1", 1, 1, 1)

        edges = [HyperEdge(joint0, joint1)]

        if partitioning == "2x1":
            # isolate upper and lower body
            parts = [(joint0,), (joint1,)]
            # TODO: There could be tons of decompositions here

        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges

    elif label in ["Swimmer-v2"]:

        # define Mujoco-Graph
        joint0 = Node("rot2", 0, 0, 0) # TODO: double-check ids
        joint1 = Node("rot3", 1, 1, 1)

        edges = [HyperEdge(joint0, joint1)]

        if partitioning == "2x1":
            # isolate upper and lower body
            parts = [(joint0,), (joint1,)]
            # TODO: There could be tons of decompositions here

        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges

    elif label in ["Walker2d-v2"]:

        # define Mujoco-Graph
        thigh_joint = Node("thigh_joint", 0, 0, 0)
        leg_joint = Node("leg_joint", 1, 1, 1)
        foot_joint = Node("foot_joint", 2, 2, 2)
        thigh_left_joint = Node("thigh_left_joint", 3, 3, 3)
        leg_left_joint = Node("leg_left_joint", 4, 4, 4)
        foot_left_joint = Node("foot_left_joint", 5, 5, 5)

        edges = [HyperEdge(foot_joint, leg_joint),
                 HyperEdge(leg_joint, thigh_joint),
                 HyperEdge(foot_left_joint, leg_left_joint),
                 HyperEdge(leg_left_joint, thigh_left_joint),
                 HyperEdge(thigh_joint, thigh_left_joint)
                 ]

        if partitioning == "2x3":
            # isolate upper and lower body
            parts = [(foot_joint, leg_joint, thigh_joint),
                     (foot_left_joint, leg_left_joint, thigh_left_joint,)]
            # TODO: There could be tons of decompositions here

        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges