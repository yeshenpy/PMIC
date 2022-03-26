import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.collaborative = True  # whether agents share rewards
        # add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(3)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want other agent to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        world.agents[1].goal_a = world.agents[0]
        world.agents[1].goal_b = np.random.choice(world.landmarks)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])               
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.75,0.25,0.25]) 
        world.landmarks[1].color = np.array([0.25,0.75,0.25]) 
        world.landmarks[2].color = np.array([0.25,0.25,0.75]) 
        # special colors for goals
        
        
        world.agents[0].goal_a.color = world.agents[0].goal_b.color                
        world.agents[1].goal_a.color = world.agents[1].goal_b.color                               
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
    
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    
        
       
    def reward(self, agent, world):
        if agent.goal_a is None or agent.goal_b is None:
            return 0.0
        #dist2 = np.sum(np.square(agent.goal_a.state.p_pos - agent.goal_b.state.p_pos))
        
        
#        one_col = False 
#        
#        
#        for l in world.landmarks:
#            if self.is_collision(world.agents[0], l):
#                one_col = True
#            
#        two_col = False 
#        for l in world.landmarks:
#            if self.is_collision(world.agents[1], l):
#                two_col = True
#        
#
#        
#        if self.is_collision(world.agents[0].goal_a, world.agents[0].goal_b):
#            two_col_target = True 
#        else :
#            two_col_target = False
#            
#        if self.is_collision(world.agents[1].goal_a, world.agents[1].goal_b):
#            one_col_target = True 
#        else :
#            one_col_target = False
#        
#        reward = 0 
#        if one_col :
#            if one_col_target:
#                reward +=20
#            else :
#                reward -=10
#            
#        if two_col :
#            if two_col_target:
#                reward +=20
#            else :
#                reward -=10
#        
#        return reward/2.0
        
        if self.is_collision(world.agents[0].goal_a, world.agents[0].goal_b) and self.is_collision(world.agents[1].goal_a, world.agents[1].goal_b):
            return 10
        elif self.is_collision(world.agents[0].goal_a, world.agents[0].goal_b) is False and self.is_collision(world.agents[1].goal_a, world.agents[1].goal_b) is True:
            return 5
        elif self.is_collision(world.agents[0].goal_a, world.agents[0].goal_b) is True and self.is_collision(world.agents[1].goal_a, world.agents[1].goal_b) is False :
            return 5 
        else :
            return 0 
            
#        if agent.collide:
#            if self.is_collision(agent.goal_a, agent.goal_b):
#                rew = 10
#            else:
#                rew = 0
#                
#        return rew

    def observation(self, agent, world):
        # goal color
        goal_color = [np.zeros(world.dim_color), np.zeros(world.dim_color)]
        if agent.goal_b is not None:
            goal_color[1] = agent.goal_b.color 

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
        return np.concatenate([agent.state.p_vel] + entity_pos + [goal_color[1]] + comm)
            