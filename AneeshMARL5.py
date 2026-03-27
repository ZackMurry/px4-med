import random
import math
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import json
from datetime import datetime
import heapq
import time


# ==================== CONFIGURATION ====================
GRID_SIZE = 50
CELL_SIZE = 20
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

NUM_OBSTACLES = 200
NUM_WIND_ZONES = 15
NUM_LOW_SIGNAL_ZONES = 10
WIND_APPEAR_INTERVAL = 30
LOW_SIGNAL_APPEAR_INTERVAL = 30

MAX_PATIENT_TIMER = 250

NUM_PATIENTS = 4
MAX_PATIENTS = 8
NEW_PATIENT_SPAWN_INTERVAL = 75
LEVEL1_PATIENT_WEIGHT = 1
LEVEL2_PATIENT_WEIGHT = 2
LEVEL3_PATIENT_WEIGHT = 3
MAX_PATIENT_WEIGHT = 3

# Training hyperparameters
NUM_EPISODES = 12000      
MAX_STEPS = 800
MAX_BATTERY = 100
BATTERY_DRAIN_PER_STEP = 0.1
BATTERY_DRAIN_IN_WIND = 0.5
LOW_BATTERY_THRESHOLD = 20
LOW_SIGNAL_FAILURE_PROB = 0.3

LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 128
BUFFER_CAPACITY = 50000   
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_FRAC = 0.95 
TARGET_UPDATE_FREQ = 10

# Reward parameters
GOAL_REWARD = 100.0           
STEP_PENALTY = -0.2           
CLEAN_STEP_BONUS = 0.1        
COLLISION_PENALTY = -1000.0
AGENT_COLLISION_PENALTY = -1000.0  
BATTERY_DEPLETION_PENALTY = -50.0
LOW_BATTERY_PENALTY = -0.5
WIND_PENALTY = -2.0
LOW_SIGNAL_PENALTY = -8.0
SHAPING_FACTOR = 1.5
PATIENT_DEATH_PENALTY = -30.0
LANDING_REWARD = 150.0       
LAND_WRONG_PENALTY = -2.0
CLOSENESS_PENALTY = -10.0
CLOSENESS_RADIUS  = 4


# ==================== ENVIRONMENT ====================
class Environment:

    def __init__(self, fixed_layout):
        self.grid_size = GRID_SIZE
        self.cell_size = CELL_SIZE
        self.fixed_layout = fixed_layout

        # Obstacle generation
        self.obstacles = set(self.generate_obstacles())

        # Fixed starting layout (used when fixed_layout=True)
        self.start_positions = [(1, 1), (1, 13)]
        self.patient_positions = [(13, 13), (13, 1), (25, 25), (25, 1),
                                  (35, 10), (10, 35), (40, 30), (30, 40)]
        self.landing_zones = [(48, 48), (48, 45)]
        self.agents = list(self.start_positions)
        self.batteries = [MAX_BATTERY, MAX_BATTERY]
        self.landed = [False, False]

        self.patients_delivered = [False] * MAX_PATIENTS
        self.patients_actually_delivered = [False] * MAX_PATIENTS
        self.patient_timers = [MAX_PATIENT_TIMER] * MAX_PATIENTS
        self.patient_weights = [self.random_weight() for _ in range(MAX_PATIENTS)]
        self.patient_active = [i < NUM_PATIENTS for i in range(MAX_PATIENTS)]
        self.new_patient_timer = NEW_PATIENT_SPAWN_INTERVAL

        # Dynamic hazard zones
        self.wind_zones = []
        self.wind_timer = WIND_APPEAR_INTERVAL
        self.low_signal_zones = []
        self.low_signal_timer = LOW_SIGNAL_APPEAR_INTERVAL
 
        self.astar_paths = self.compute_astar_paths()


    def random_weight(self):
        x = random.randint(1, 3)
        if x == 1: return LEVEL1_PATIENT_WEIGHT
        if x == 2: return LEVEL2_PATIENT_WEIGHT
        return LEVEL3_PATIENT_WEIGHT

    def manhattan_distance(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def direction_vector(self, pos, target):
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        dist = math.sqrt(dx**2 + dy**2)
        if dist > 0:
            return dx / dist, dy / dist
        return 0.0, 0.0

    def nearest_undelivered_patient(self, pos):
        best_idx = None
        best_dist = float('inf')
        for p in range(MAX_PATIENTS):
            if not self.patient_active[p]:
                continue
            if not self.patients_delivered[p]:
                d = self.manhattan_distance(pos, self.patient_positions[p])
                if d < best_dist:
                    best_dist = d
                    best_idx = p
        return best_idx

    def generate_obstacles(self):
        protected = {(1,1),(1,13),(13,13),(13,1),(25,25),(25,1),(48,48)}
        obstacles = set()

        for y in range(3, 12):
            if y not in [5, 6, 9, 10]:
                obstacles.add((7, y))

        while len(obstacles) < NUM_OBSTACLES:
            x = random.randint(2, self.grid_size - 3)
            y = random.randint(2, self.grid_size - 3)
            if (x, y) not in protected:
                neighbors = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
                blocked = sum(
                    1 for n in neighbors
                    if n in obstacles or n[0] < 0 or n[0] >= self.grid_size
                       or n[1] < 0 or n[1] >= self.grid_size
                )
                if blocked < 3:
                    obstacles.add((x, y))

        return list(obstacles)

    def a_star(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score   = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            x, y = current
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nb = (x+dx, y+dy)
                if not (0 <= nb[0] < self.grid_size and 0 <= nb[1] < self.grid_size):
                    continue
                if nb in self.obstacles:
                    continue
                tg = g_score[current] + 1
                if nb not in g_score or tg < g_score[nb]:
                    came_from[nb] = current
                    g_score[nb]   = tg
                    heapq.heappush(open_set, (tg + self.manhattan_distance(nb, goal), nb))
        return []

    def compute_astar_paths(self):
        paths = []
        for ag_pos in self.agents:
            for pat_pos in self.patient_positions:
                path = self.a_star(ag_pos, pat_pos)
                if path:
                    paths.append(path)
        return paths

    def update_wind_zones(self):
        if self.wind_timer > 0:
            self.wind_timer -= 1
            return
        candidates = list({pos for path in self.astar_paths for pos in path})
        self.wind_zones = (
            random.sample(candidates, NUM_WIND_ZONES)
            if len(candidates) >= NUM_WIND_ZONES
            else candidates
        )
        self.wind_timer = WIND_APPEAR_INTERVAL

    def update_low_signal_zones(self):
        if self.low_signal_timer > 0:
            self.low_signal_timer -= 1
            return
        candidates = list({pos for path in self.astar_paths for pos in path})
        self.low_signal_zones = (
            random.sample(candidates, NUM_LOW_SIGNAL_ZONES)
            if len(candidates) >= NUM_LOW_SIGNAL_ZONES
            else candidates
        )
        self.low_signal_timer = LOW_SIGNAL_APPEAR_INTERVAL

    def get_state(self, agent_idx):

        pos = self.agents[agent_idx]
        other_pos = self.agents[1 - agent_idx]
        x, y = pos

        agent_id = float(agent_idx)

        x_norm = x / self.grid_size
        y_norm = y / self.grid_size
        bat_norm = self.batteries[agent_idx] / MAX_BATTERY
        landed_f = 1.0 if self.landed[agent_idx] else 0.0

        other_dx = (other_pos[0] - x) / self.grid_size
        other_dy = (other_pos[1] - y) / self.grid_size

        lz_dir_x, lz_dir_y = self.direction_vector(pos, self.landing_zones[agent_idx])

        patient_features = []
        #   not yet spawned : [0, 0, 0, 0, 0, 0]  (all zeros)
        #   delivered / dead: [0, 0, 0, 0, 1, 0]  (delivered_flag=1)
        #   active          : [gx, gy, dx, dy, 0, timer_norm]
        for p in range(MAX_PATIENTS):
            if not self.patient_active[p]:
                patient_features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.patient_weights[p]/MAX_PATIENT_WEIGHT])
            elif self.patients_delivered[p]:
                patient_features.extend([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, self.patient_weights[p]/MAX_PATIENT_WEIGHT])
            else:
                pp = self.patient_positions[p]
                gx_norm = pp[0] / self.grid_size
                gy_norm = pp[1] / self.grid_size
                dir_x, dir_y = self.direction_vector(pos, pp)
                timer_norm = self.patient_timers[p] / MAX_PATIENT_TIMER
                patient_features.extend([gx_norm, gy_norm, dir_x, dir_y, 0.0, timer_norm, self.patient_weights[p]/MAX_PATIENT_WEIGHT])

        # 5x5 local view
        obs_grid = []
        wind_grid = []
        ls_grid = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                cx, cy = x + dx, y + dy
                out_of_bounds = (cx < 0 or cx >= self.grid_size or
                                 cy < 0 or cy >= self.grid_size)
                obs_grid.append( 1.0 if out_of_bounds or (cx, cy) in self.obstacles       else 0.0)
                wind_grid.append(1.0 if not out_of_bounds and (cx, cy) in self.wind_zones  else 0.0)
                ls_grid.append(  1.0 if not out_of_bounds and (cx, cy) in self.low_signal_zones else 0.0)

        state = (
            [agent_id,                                                                          
             x_norm, y_norm, bat_norm, landed_f, other_dx, other_dy, lz_dir_x, lz_dir_y]      
            + patient_features                                                                   
            + obs_grid                                                                           
            + wind_grid                                                                          
            + ls_grid                                                                            
        )  

        return state


    def step(self, actions):
        self.update_wind_zones()
        self.update_low_signal_zones()

        step_data = {
            'wind_entries':       [0, 0],
            'low_signal_entries': [0, 0],
            'obstacle_collisions': 0,
            'agent_collisions':    0
        }

        old_positions = [tuple(pos) for pos in self.agents]

        old_shaping_dist = []
        for i in range(2):
            np_idx = self.nearest_undelivered_patient(old_positions[i])
            if np_idx is not None:
                old_shaping_dist.append(
                    self.manhattan_distance(old_positions[i], self.patient_positions[np_idx])
                )
            else:
                old_shaping_dist.append(
                    self.manhattan_distance(old_positions[i], self.landing_zones[i])
                )

        step_rewards = [STEP_PENALTY, STEP_PENALTY]
        milestone_rewards = [0.0, 0.0]

        new_positions = []
        for agent_idx, action in enumerate(actions):
            x, y = self.agents[agent_idx]

            if self.landed[agent_idx]:
                new_positions.append((x, y))
                continue

            if action == 4:
                if (x, y) == self.landing_zones[agent_idx]:
                    self.landed[agent_idx] = True
                    milestone_rewards[agent_idx] += LANDING_REWARD
                else:
                    step_rewards[agent_idx] += LAND_WRONG_PENALTY
                new_positions.append((x, y))
                continue

            in_low_signal = (x, y) in self.low_signal_zones
            if in_low_signal and random.random() < LOW_SIGNAL_FAILURE_PROB:
                new_x, new_y = x, y 
            else:
                if   action == 0: new_x, new_y = x,     y - 1
                elif action == 1: new_x, new_y = x,     y + 1
                elif action == 2: new_x, new_y = x - 1, y
                elif action == 3: new_x, new_y = x + 1, y
                else:             new_x, new_y = x,     y

            if new_x < 0 or new_x >= self.grid_size or new_y < 0 or new_y >= self.grid_size:
                milestone_rewards[agent_idx] += COLLISION_PENALTY
                step_data['obstacle_collisions'] += 1
                new_x, new_y = x, y
            elif (new_x, new_y) in self.obstacles:
                milestone_rewards[agent_idx] += COLLISION_PENALTY
                step_data['obstacle_collisions'] += 1
                new_x, new_y = x, y

            new_positions.append((new_x, new_y))

        active = [i for i in range(2) if not self.landed[i]]
        if (len(active) == 2
                and new_positions[active[0]] == new_positions[active[1]]):
            for i in active:
                milestone_rewards[i] += AGENT_COLLISION_PENALTY
            step_data['agent_collisions'] += 1
            for i in active:
                new_positions[i] = old_positions[i]

        self.agents = new_positions

        active_agents = [i for i in range(2) if not self.landed[i]]
        if len(active_agents) == 2:
            dist = (abs(self.agents[0][0] - self.agents[1][0]) +
                    abs(self.agents[0][1] - self.agents[1][1]))
            if dist < CLOSENESS_RADIUS:
                for i in active_agents:
                    step_rewards[i] += CLOSENESS_PENALTY

        for i in range(2):
            if self.landed[i]:
                continue

            in_wind = self.agents[i] in self.wind_zones
            if in_wind:
                self.batteries[i] -= (BATTERY_DRAIN_PER_STEP + BATTERY_DRAIN_IN_WIND)
                step_rewards[i] += WIND_PENALTY
                step_data['wind_entries'][i] += 1
            else:
                self.batteries[i] -= BATTERY_DRAIN_PER_STEP

            if self.agents[i] in self.low_signal_zones:
                step_rewards[i] += LOW_SIGNAL_PENALTY
                step_data['low_signal_entries'][i] += 1

            if 0 < self.batteries[i] < LOW_BATTERY_THRESHOLD:
                step_rewards[i] += LOW_BATTERY_PENALTY

            if self.batteries[i] <= 0:
                milestone_rewards[i] += BATTERY_DEPLETION_PENALTY 
                self.batteries[i] = 0

        self.new_patient_timer -= 1
        if self.new_patient_timer <= 0:
            self.new_patient_timer = NEW_PATIENT_SPAWN_INTERVAL
            for p in range(MAX_PATIENTS):
                if not self.patient_active[p] and not self.patients_delivered[p]:
                    self.patient_active[p] = True
                    self.patient_timers[p] = MAX_PATIENT_TIMER
                    self.patient_weights[p] = self.random_weight()
                    break

        for p in range(MAX_PATIENTS):
            if not self.patient_active[p]:
                continue
            if not self.patients_delivered[p]:
                self.patient_timers[p] -= 1
                if self.patient_timers[p] <= 0:
                    milestone_rewards[0] += PATIENT_DEATH_PENALTY / 2  
                    milestone_rewards[1] += PATIENT_DEATH_PENALTY / 2
                    self.patients_delivered[p] = True

        for i in range(2):
            if self.landed[i]:
                continue
            for p in range(MAX_PATIENTS):
                if not self.patient_active[p]:
                    continue
                if (not self.patients_delivered[p]
                        and self.agents[i] == self.patient_positions[p]):
                    timer_ratio = self.patient_timers[p] / MAX_PATIENT_TIMER
                    # TODO: Check if this is the correct way to do triage. Maybe have it as a eperate value?
                    delivery_reward = GOAL_REWARD * timer_ratio * self.patient_weights[p]
                    milestone_rewards[i] += delivery_reward             
                    self.patients_delivered[p] = True
                    self.patients_actually_delivered[p] = True

        for i in range(2):
            if self.landed[i]:
                continue
            np_idx = self.nearest_undelivered_patient(self.agents[i])
            if np_idx is not None:
                new_dist = self.manhattan_distance(
                    self.agents[i], self.patient_positions[np_idx]
                )
            else:
                new_dist = self.manhattan_distance(self.agents[i], self.landing_zones[i])
            step_rewards[i] += SHAPING_FACTOR * (old_shaping_dist[i] - new_dist)

        STEP_CLIP = 5.0
        rewards = [
            max(-STEP_CLIP, min(STEP_CLIP, step_rewards[i])) + milestone_rewards[i]
            for i in range(2)
        ]

        battery_depleted = [self.batteries[i] <= 0 for i in range(2)]

        done = all(self.landed) or any(battery_depleted)

        next_states = [self.get_state(i) for i in range(2)]
        return next_states, rewards, done, step_data

  
    def generate_random_positions(self):
        MIN_DIST = 6
        positions = []
        max_attempts = 300

        total_needed = 2 + MAX_PATIENTS + 2

        while len(positions) < total_needed:
            placed = False
            for _ in range(max_attempts):
                x = random.randint(1, self.grid_size - 2)
                y = random.randint(1, self.grid_size - 2)
                pos = (x, y)
                if pos in self.obstacles:
                    continue
                if any(abs(pos[0]-ep[0]) + abs(pos[1]-ep[1]) < MIN_DIST
                       for ep in positions):
                    continue
                positions.append(pos)
                placed = True
                break

            if not placed:
                fallbacks = [
                    (1, 1), (1, self.grid_size-2),
                    (self.grid_size-2, 1), (self.grid_size-2, self.grid_size-2),
                    (self.grid_size//2, self.grid_size//2),
                    (self.grid_size//4, 3*self.grid_size//4),
                    (3*self.grid_size//4, self.grid_size//4),
                    (self.grid_size//3, self.grid_size//3),
                    (2*self.grid_size//3, 2*self.grid_size//3),
                    (self.grid_size//4, self.grid_size//4),
                ]
                positions.append(fallbacks[len(positions) % len(fallbacks)])

        self.start_positions = [positions[0], positions[1]]
        self.patient_positions = positions[2:2 + MAX_PATIENTS]
        self.landing_zones = [positions[2 + MAX_PATIENTS], positions[2 + MAX_PATIENTS + 1]]

    def reset(self):
        if not self.fixed_layout:
            self.generate_random_positions()

        self.agents = list(self.start_positions)
        self.batteries = [MAX_BATTERY, MAX_BATTERY]
        self.landed = [False, False]

        self.patients_delivered = [False] * MAX_PATIENTS
        self.patients_actually_delivered = [False] * MAX_PATIENTS
        self.patient_timers = [MAX_PATIENT_TIMER] * MAX_PATIENTS
        self.patient_weights = [self.random_weight() for _ in range(MAX_PATIENTS)]
        self.patient_active = [i < NUM_PATIENTS for i in range(MAX_PATIENTS)]
        self.new_patient_timer = NEW_PATIENT_SPAWN_INTERVAL

        self.wind_zones = []
        self.wind_timer = WIND_APPEAR_INTERVAL
        self.low_signal_zones = []
        self.low_signal_timer = LOW_SIGNAL_APPEAR_INTERVAL

        self.astar_paths = self.compute_astar_paths()

        return [self.get_state(i) for i in range(2)]

    def render(self, screen):
        screen.fill((255, 255, 255))
        font_small = pygame.font.SysFont("arial", 7)

        # Grid lines
        for x in range(0, WINDOW_SIZE, self.cell_size):
            pygame.draw.line(screen, (200,200,200), (x,0), (x,WINDOW_SIZE))
        for y in range(0, WINDOW_SIZE, self.cell_size):
            pygame.draw.line(screen, (200,200,200), (0,y), (WINDOW_SIZE,y))

        # Obstacles
        for obs in self.obstacles:
            pygame.draw.rect(screen, (0,0,0),
                pygame.Rect(obs[0]*self.cell_size, obs[1]*self.cell_size,
                            self.cell_size, self.cell_size))

        # Wind zones
        for wz in self.wind_zones:
            pygame.draw.rect(screen, (255,165,0),
                pygame.Rect(wz[0]*self.cell_size, wz[1]*self.cell_size,
                            self.cell_size, self.cell_size))

        # Low signal zones
        for lsz in self.low_signal_zones:
            pygame.draw.rect(screen, (138,43,226),
                pygame.Rect(lsz[0]*self.cell_size, lsz[1]*self.cell_size,
                            self.cell_size, self.cell_size))

        # Landing Zones
        zone_colors = [(200, 255, 200), (200, 200, 255)]
        for idx, lz in enumerate(self.landing_zones):
            pygame.draw.rect(screen, zone_colors[idx],
                pygame.Rect(lz[0]*self.cell_size, lz[1]*self.cell_size,
                            self.cell_size, self.cell_size))

        # Patients 
        for p in range(MAX_PATIENTS):
            if not self.patient_active[p]:
                continue
            pp = self.patient_positions[p]
            if self.patients_delivered[p]:
                color = (180, 180, 180)   
            else:
                ratio = self.patient_timers[p] / MAX_PATIENT_TIMER
                if ratio > 0.6:
                    color = (255, 120, 120)   
                elif ratio > 0.3:
                    color = (255, 200, 50)    
                else:
                    color = (220, 0, 0)      
            pygame.draw.rect(screen, color,
                pygame.Rect(pp[0]*self.cell_size, pp[1]*self.cell_size,
                            self.cell_size, self.cell_size))
            
            if not self.patients_delivered[p]:
                surf = font_small.render(str(self.patient_timers[p]), True, (0,0,0))
                screen.blit(surf, (pp[0]*self.cell_size, pp[1]*self.cell_size))

        # Agents
        agent_colors = [(200,0,0), (0,0,200)]
        landed_colors = [(80,0,0),  (0,0,80)]
        for idx, pos in enumerate(self.agents):
            color = landed_colors[idx] if self.landed[idx] else agent_colors[idx]
            center = (pos[0]*self.cell_size + self.cell_size//2,
                      pos[1]*self.cell_size + self.cell_size//2)
            pygame.draw.circle(screen, color, center, self.cell_size//3)
            pygame.draw.circle(screen, (255,255,255), center, self.cell_size//3, 2)

            # Battery bar
            bpct = max(0.0, self.batteries[idx] / MAX_BATTERY)
            bw = self.cell_size - 2
            bh = 3
            bx = pos[0]*self.cell_size + 1
            by = pos[1]*self.cell_size - 5
            pygame.draw.rect(screen, (100,100,100), (bx, by, bw, bh))
            if bpct > 0:
                bc = (0,255,0) if bpct > 0.5 else ((255,255,0) if bpct > 0.2 else (255,0,0))
                pygame.draw.rect(screen, bc, (bx, by, int(bw*bpct), bh))

        pygame.display.flip()


# ==================== CENTRAL Q-NETWORK (CTDE) ====================
class CentralQNet(nn.Module):
    def __init__(self, joint_state_dim, action_dim):
        super(CentralQNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(joint_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.network(x)


# ==================== JOINT REPLAY BUFFER (CTDE) ====================
class JointReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, s0, s1, a0, a1, r0, r1, ns0, ns1, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (s0, s1, a0, a1, r0, r1, ns0, ns1, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class CTDEAgent:
    def __init__(self, state_dim, action_dim, lr, gamma):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.state_dim = state_dim
        joint_dim = state_dim * 2   

        self.policy_net = CentralQNet(joint_dim, action_dim).to(self.device)
        self.target_net = CentralQNet(joint_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.joint_buffer = JointReplayBuffer(BUFFER_CAPACITY)

        self.recent_losses = []
        self.recent_q_values = []

    def select_action(self, s_self, s_other, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        joint = torch.FloatTensor(s_self + s_other).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.policy_net(joint)
            self.recent_q_values.append(q_vals.max().item())
        return q_vals.argmax().item()

    def push(self, s0, s1, a0, a1, r0, r1, ns0, ns1, done):

        self.joint_buffer.push(s0, s1, a0, a1, r0, r1, ns0, ns1, done)

    def train_step(self, batch_size):
        if len(self.joint_buffer) < batch_size:
            return None

        batch = self.joint_buffer.sample(batch_size)
        s0, s1, a0, a1, r0, r1, ns0, ns1, dones = zip(*batch)

        # Convert to tensors
        S0 = torch.FloatTensor(s0).to(self.device)
        S1 = torch.FloatTensor(s1).to(self.device)
        A0 = torch.LongTensor(a0).unsqueeze(1).to(self.device)
        A1 = torch.LongTensor(a1).unsqueeze(1).to(self.device)
        R0 = torch.FloatTensor(r0).unsqueeze(1).to(self.device)
        R1 = torch.FloatTensor(r1).unsqueeze(1).to(self.device)
        NS0 = torch.FloatTensor(ns0).to(self.device)
        NS1 = torch.FloatTensor(ns1).to(self.device)
        DONE = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Joint inputs
        joint_s_ag0 = torch.cat([S0,  S1],  dim=1)   
        joint_s_ag1 = torch.cat([S1,  S0],  dim=1)   
        joint_ns_ag0 = torch.cat([NS0, NS1], dim=1)
        joint_ns_ag1 = torch.cat([NS1, NS0], dim=1)

        # Current Q values
        q_ag0 = self.policy_net(joint_s_ag0).gather(1, A0)
        q_ag1 = self.policy_net(joint_s_ag1).gather(1, A1)

        # Target Q values
        with torch.no_grad():
            next_q_ag0 = self.target_net(joint_ns_ag0).max(1)[0].unsqueeze(1)
            next_q_ag1 = self.target_net(joint_ns_ag1).max(1)[0].unsqueeze(1)

        target_ag0 = R0 + self.gamma * next_q_ag0 * (1 - DONE)
        target_ag1 = R1 + self.gamma * next_q_ag1 * (1 - DONE)

        loss = nn.MSELoss()(q_ag0, target_ag0) + nn.MSELoss()(q_ag1, target_ag1)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.recent_losses.append(loss.item())
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_metrics(self):
        avg_loss = np.mean(self.recent_losses)   if self.recent_losses   else 0.0
        avg_q = np.mean(self.recent_q_values) if self.recent_q_values else 0.0
        self.recent_losses = []
        self.recent_q_values = []
        return avg_loss, avg_q


# ==================== DATA COLLECTOR ====================
class Data_Collection:
    def __init__(self):
        self.episodes                  = []
        self.total_rewards             = []
        self.success_rate              = []          
        self.agent_0_delivered         = []
        self.agent_1_delivered         = []
        self.patients_delivered_counts = []         
        self.patients_died_counts      = []
        self.patients_spawned_counts   = []
        self.agent_0_landed            = []          
        self.agent_1_landed            = []          
        self.steps_per_episode         = []
        self.collisions_obstacles      = []
        self.collisions_agents         = []
        self.wind_entries_agent0       = []
        self.wind_entries_agent1       = []
        self.low_signal_entries_agent0 = []
        self.low_signal_entries_agent1 = []
        self.battery_remaining_agent0  = []
        self.battery_remaining_agent1  = []
        self.epsilon_values            = []

        # --- TRIAGE / FAIRNESS DATA ---
        # Deliveries and deaths broken down by patient weight class so we can
        # measure whether the agents learned to prioritise high-acuity patients.
        self.delivered_w1              = []   # count of weight-1 patients delivered
        self.delivered_w2              = []   # count of weight-2 patients delivered
        self.delivered_w3              = []   # count of weight-3 patients delivered
        self.died_w1                   = []   # count of weight-1 patients that timed out
        self.died_w2                   = []
        self.died_w3                   = []
        # weighted_delivery_score = sum of weights for every actually-delivered patient
        # max_possible_weighted_score  = sum of weights for every patient that spawned
        # triage_efficiency = weighted_score / max_possible  (1.0 = perfect prioritisation)
        self.weighted_delivery_score   = []
        self.max_possible_weighted_score = []
        self.triage_efficiency         = []

    def log_episode(self, episode, total_reward, success,
                    agent_delivered, patients_delivered_count, patients_died_count,
                    patients_spawned_count, landed, steps, collisions_obs, collisions_ag,
                    wind_entries, low_signal_entries, epsilon, batteries,
                    triage_data):
        self.episodes.append(episode)
        self.total_rewards.append(total_reward)
        self.success_rate.append(1 if success else 0)
        self.agent_0_delivered.append(1 if agent_delivered[0] else 0)
        self.agent_1_delivered.append(1 if agent_delivered[1] else 0)
        self.patients_delivered_counts.append(patients_delivered_count)
        self.patients_died_counts.append(patients_died_count)
        self.patients_spawned_counts.append(patients_spawned_count)
        self.agent_0_landed.append(1 if landed[0] else 0)
        self.agent_1_landed.append(1 if landed[1] else 0)
        self.steps_per_episode.append(steps)
        self.collisions_obstacles.append(collisions_obs)
        self.collisions_agents.append(collisions_ag)
        self.wind_entries_agent0.append(wind_entries[0])
        self.wind_entries_agent1.append(wind_entries[1])
        self.low_signal_entries_agent0.append(low_signal_entries[0])
        self.low_signal_entries_agent1.append(low_signal_entries[1])
        self.battery_remaining_agent0.append(batteries[0])
        self.battery_remaining_agent1.append(batteries[1])
        self.epsilon_values.append(epsilon)
        self.delivered_w1.append(triage_data['delivered_w1'])
        self.delivered_w2.append(triage_data['delivered_w2'])
        self.delivered_w3.append(triage_data['delivered_w3'])
        self.died_w1.append(triage_data['died_w1'])
        self.died_w2.append(triage_data['died_w2'])
        self.died_w3.append(triage_data['died_w3'])
        self.weighted_delivery_score.append(triage_data['weighted_delivery_score'])
        self.max_possible_weighted_score.append(triage_data['max_possible_weighted_score'])
        self.triage_efficiency.append(triage_data['triage_efficiency'])

    def save_to_json(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = f"marl_training_data_{timestamp}.json"

        data = {
            'episodes':                  self.episodes,
            'total_rewards':             self.total_rewards,
            'success_rate':              self.success_rate,
            'agent_0_delivered':         self.agent_0_delivered,
            'agent_1_delivered':         self.agent_1_delivered,
            'patients_delivered_counts': self.patients_delivered_counts,
            'patients_died_counts':      self.patients_died_counts,
            'patients_spawned_counts':   self.patients_spawned_counts,
            'agent_0_landed':            self.agent_0_landed,
            'agent_1_landed':            self.agent_1_landed,
            'steps_per_episode':         self.steps_per_episode,
            'collisions_obstacles':      self.collisions_obstacles,
            'collisions_agents':         self.collisions_agents,
            'wind_entries_agent0':       self.wind_entries_agent0,
            'wind_entries_agent1':       self.wind_entries_agent1,
            'low_signal_entries_agent0': self.low_signal_entries_agent0,
            'low_signal_entries_agent1': self.low_signal_entries_agent1,
            'battery_remaining_agent0':  self.battery_remaining_agent0,
            'battery_remaining_agent1':  self.battery_remaining_agent1,
            'epsilon_values':            self.epsilon_values,
            'delivered_w1':              self.delivered_w1,
            'delivered_w2':              self.delivered_w2,
            'delivered_w3':              self.delivered_w3,
            'died_w1':                   self.died_w1,
            'died_w2':                   self.died_w2,
            'died_w3':                   self.died_w3,
            'weighted_delivery_score':   self.weighted_delivery_score,
            'max_possible_weighted_score': self.max_possible_weighted_score,
            'triage_efficiency':         self.triage_efficiency,
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n Data saved to {filename}")
        return filename


# ==================== TRAINING ====================
def train():
    print("\n" + "="*60)
    print("MARL TRAINING  (CTDE | dynamic patient spawning | two landing zones | closeness penalty)")
    print("="*60)
    print(f"Grid Size:            {GRID_SIZE}x{GRID_SIZE}")
    print(f"Episodes:             {NUM_EPISODES}")
    print(f"Max Steps:            {MAX_STEPS}")
    print(f"Initial Patients:     {NUM_PATIENTS}")
    print(f"Max Patients:         {MAX_PATIENTS}")
    print(f"Spawn Interval:       {NEW_PATIENT_SPAWN_INTERVAL} steps")
    print(f"Patient Timer:        {MAX_PATIENT_TIMER} steps")
    print(f"Learning Rate:        {LEARNING_RATE}")
    print(f"Obstacles:            {NUM_OBSTACLES}")
    print(f"Max Battery:          {MAX_BATTERY}")
    print(f"Wind Zones:           {NUM_WIND_ZONES}")
    print(f"Low Signal Zones:     {NUM_LOW_SIGNAL_ZONES}")
    print(f"Low Signal Fail Prob: {LOW_SIGNAL_FAILURE_PROB}")
    print("="*60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    env = Environment(fixed_layout=False)
    data = Data_Collection()
    state_dim = len(env.get_state(0))
    action_dim = 5   
    joint_dim = state_dim * 2   

    print(f"State dimension (per agent): {state_dim}")
    print(f"Joint state dimension:       {joint_dim}")
    print(f"Action dimension:            {action_dim}\n")

    ctde_agent = CTDEAgent(state_dim, action_dim, LEARNING_RATE, GAMMA)

    TRAIN_TIME_LIMIT = 5 * 60  # 5 minutes
    train_start = time.monotonic()

    epsilon = EPSILON_START
    epsilon_decay_eps = int(NUM_EPISODES * EPSILON_DECAY_FRAC)
    epsilon_decay = (EPSILON_END / EPSILON_START) ** (1 / epsilon_decay_eps)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("MARL — Medical Supply Delivery")
    clock = pygame.time.Clock()

    for episode in range(NUM_EPISODES):
        states = env.reset()
        episode_rewards = [0.0, 0.0]
        total_reward = 0.0
        steps = 0
        episode_wind_entries = [0, 0]
        episode_low_signal_entries = [0, 0]
        episode_obstacle_collisions = 0
        episode_agent_collisions = 0

        for step in range(MAX_STEPS):
            steps += 1

            a0 = ctde_agent.select_action(states[0], states[1], epsilon)
            a1 = ctde_agent.select_action(states[1], states[0], epsilon)
            actions = [a0, a1]

            next_states, rewards, done, step_data = env.step(actions)
            
            for i in range(2):
                episode_rewards[i] += rewards[i]
                episode_wind_entries[i] += step_data['wind_entries'][i]
                episode_low_signal_entries[i] += step_data['low_signal_entries'][i]
            episode_obstacle_collisions += step_data['obstacle_collisions']
            episode_agent_collisions += step_data['agent_collisions']

      
            ctde_agent.push(
                states[0], states[1],
                a0, a1,
                rewards[0], rewards[1],
                next_states[0], next_states[1],
                done
            )
            total_reward += rewards[0] + rewards[1]

            ctde_agent.train_step(BATCH_SIZE)

            states = next_states

            if episode % 800 == 0:
                env.render(screen)
                clock.tick(10)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        data.save_to_json()
                        return

            if done:
                break

        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            ctde_agent.update_target()
    
        if episode < epsilon_decay_eps:
            epsilon = max(EPSILON_END, epsilon * epsilon_decay)

        patients_delivered_count = sum(env.patients_actually_delivered)
        patients_died_count      = sum(
            1 for p in range(MAX_PATIENTS)
            if env.patients_delivered[p] and not env.patients_actually_delivered[p]
        )
        patients_spawned_count = sum(env.patient_active)

        triage_data = {
            'delivered_w1': 0, 'delivered_w2': 0, 'delivered_w3': 0,
            'died_w1':      0, 'died_w2':      0, 'died_w3':      0,
            'weighted_delivery_score':    0.0,
            'max_possible_weighted_score': 0.0,
            'triage_efficiency':          0.0,
        }
        for p in range(MAX_PATIENTS):
            if not env.patient_active[p]:
                continue
            w = int(env.patient_weights[p])
            triage_data['max_possible_weighted_score'] += w
            if env.patients_actually_delivered[p]:
                triage_data[f'delivered_w{w}'] += 1
                triage_data['weighted_delivery_score'] += w
            elif env.patients_delivered[p]:
                triage_data[f'died_w{w}'] += 1
        if triage_data['max_possible_weighted_score'] > 0:
            triage_data['triage_efficiency'] = (
                triage_data['weighted_delivery_score'] /
                triage_data['max_possible_weighted_score']
            )

        success = all(env.landed)

        data.log_episode(
            episode                  = episode,
            total_reward             = total_reward,
            success                  = success,
            agent_delivered          = env.patients_actually_delivered[:2],
            patients_delivered_count = patients_delivered_count,
            patients_died_count      = patients_died_count,
            patients_spawned_count   = patients_spawned_count,
            landed                   = env.landed,
            steps                    = steps,
            collisions_obs           = episode_obstacle_collisions,
            collisions_ag            = episode_agent_collisions,
            wind_entries             = episode_wind_entries,
            low_signal_entries       = episode_low_signal_entries,
            epsilon                  = epsilon,
            batteries                = env.batteries,
            triage_data              = triage_data,
        )

        print(
            f"Ep {episode+1:>5} | Reward: {total_reward:>8.2f} | "
            f"ε: {epsilon:.3f} | Landed: {env.landed} | "
            f"Spawned: {patients_spawned_count}/{MAX_PATIENTS} | "
            f"Delivered: {patients_delivered_count}/{patients_spawned_count} | "
            f"Died: {patients_died_count}/{patients_spawned_count} | "
            f"Steps: {steps} | "
            f"Wind: {episode_wind_entries} | LS: {episode_low_signal_entries} | "
            f"Collisions(obs/ag): {episode_obstacle_collisions}/{episode_agent_collisions}"
        )

        if episode == NUM_EPISODES - 1:
            data.save_to_json()

        if time.monotonic() - train_start >= TRAIN_TIME_LIMIT:
            print(f"\n5-minute limit reached after {episode + 1} episodes — saving model.")
            data.save_to_json()
            break

    pygame.quit()

    # Final stats
    print("\nFINAL STATISTICS")
    print("="*60)
    total_ep      = len(data.episodes)
    overall_succ  = sum(data.success_rate) / total_ep * 100
    final_succ    = sum(data.success_rate[-100:]) / min(100, total_ep) * 100
    avg_rew_all   = np.mean(data.total_rewards)
    avg_rew_final = np.mean(data.total_rewards[-100:])
    avg_patients  = np.mean(data.patients_delivered_counts)
    avg_died      = np.mean(data.patients_died_counts)

    print(f"Total Episodes:                {total_ep}")
    print(f"Overall Both-Landed Rate:      {overall_succ:.2f}%")
    print(f"Final 100 Both-Landed Rate:    {final_succ:.2f}%")
    print(f"Average Reward (All):          {avg_rew_all:.2f}")
    print(f"Average Reward (Final 100):    {avg_rew_final:.2f}")
    print(f"Avg Patients Delivered/Ep:     {avg_patients:.2f}")
    print(f"Avg Patients Died/Ep:          {avg_died:.2f}")
    print(f"Total Wind Entries:            {sum(data.wind_entries_agent0)+sum(data.wind_entries_agent1)}")
    print(f"Total Low Signal Entries:      {sum(data.low_signal_entries_agent0)+sum(data.low_signal_entries_agent1)}")
    print(f"Total Collisions (obs/ag):     {sum(data.collisions_obstacles)}/{sum(data.collisions_agents)}")
    print("="*60)

    torch.save(ctde_agent.policy_net.state_dict(), "ctde_agent_marl9.pth")
    print("\nCTDE model saved: ctde_agent_marl9.pth")
    print("="*60 + "\n")


if __name__ == '__main__':
    train()