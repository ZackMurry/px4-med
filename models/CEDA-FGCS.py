#!/usr/bin/env python3
"""CEDA-FGCS-PX4 inference policy."""

from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


# Model configuration
PACKAGE_NAME = "CEDA-FGCS-PX4"
MODEL_VERSION = "CEDA-FGCS-PX4"
CHECKPOINT_FILENAME = "ctde_agent_marl_FGCS.pth"

GRID_SIZE = 100
NUM_AGENTS = 5
MAX_PATIENTS = 50
LOCAL_GRID_RADIUS = 10
LOCAL_GRID_SIZE = 2 * LOCAL_GRID_RADIUS + 1
MAX_PATIENT_TIMER = 300
CLOSENESS_RADIUS = 4

MAX_BATTERY = 100.0
BATTERY_DRAIN_PER_STEP = 0.20
BATTERY_DRAIN_IN_WIND = 2.30
BATTERY_DRAIN_AT_LANDING_ZONE = 0.02
SAFE_RETURN_BATTERY_BUFFER = 18.0
RETURN_ENERGY_RISK_MULTIPLIER = 1.0

ACTION_NORTH = 0
ACTION_SOUTH = 1
ACTION_WEST = 2
ACTION_EAST = 3
ACTION_HOVER = 4
ACTION_LAND = 5
ACTION_DIM = 6
ACTION_NAMES = ("north", "south", "west", "east", "hover", "land")
MOVEMENT_OFFSETS = ((0, -1), (0, 1), (-1, 0), (1, 0))

DRONE_STATE_DIM = 16 + ACTION_DIM
PATIENT_STATE_DIM = 10
MISSION_STATE_DIM = 12


# Network dimensions
ENTITY_EMBED_DIM = 64
ATTENTION_HEADS = 4
SET_ATTENTION_BLOCKS = 2
SELF_EMBED_DIM = 128
GRID_EMBED_DIM = 64
AGENT_ID_EMBED_DIM = 16
CENTRAL_GRID_EMBED_DIM = 32
MIXER_EMBED_DIM = 64
GLOBAL_EMBED_DIM = 128
AGENT_MIX_CONTEXT_DIM = 128
MIXER_MIN_RAW_WEIGHT = 0.10

# Entity-attention encoders
class SetAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SetAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, entities, valid_mask):
        attention_output, _ = self.attention(
            entities,
            entities,
            entities,
            key_padding_mask=~valid_mask,
            need_weights=False
        )
        entities = self.norm1(entities + attention_output)
        entities = self.norm2(entities + self.feed_forward(entities))
        return entities * valid_mask.unsqueeze(-1).float()


class EntitySetTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_blocks):
        super(EntitySetTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.blocks = nn.ModuleList([
            SetAttentionBlock(embed_dim, num_heads)
            for _ in range(num_blocks)
        ])
        self.pool_seed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.pool_seed, std=0.02)
        self.pool_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.pool_norm1 = nn.LayerNorm(embed_dim)
        self.pool_feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.pool_norm2 = nn.LayerNorm(embed_dim)
        self.null_entity = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, entities, valid_mask):
        leading_shape = entities.shape[:-2]
        set_size = entities.shape[-2]
        entities = entities.reshape(-1, set_size, entities.shape[-1])
        valid_mask = valid_mask.reshape(-1, set_size).bool()

        encoded = self.input_projection(entities)
        empty_sets = ~valid_mask.any(dim=1, keepdim=True)
        null_entities = self.null_entity.expand(encoded.shape[0], -1, -1)
        encoded = torch.cat([encoded, null_entities], dim=1)
        valid_mask = torch.cat([valid_mask, empty_sets], dim=1)

        for block in self.blocks:
            encoded = block(encoded, valid_mask)

        seed = self.pool_seed.expand(encoded.shape[0], -1, -1)
        pooled, _ = self.pool_attention(
            seed,
            encoded,
            encoded,
            key_padding_mask=~valid_mask,
            need_weights=False
        )
        pooled = self.pool_norm1(seed + pooled)
        pooled = self.pool_norm2(pooled + self.pool_feed_forward(pooled))
        pooled = pooled.squeeze(1)

        pooled = pooled * (~empty_sets).float()
        return pooled.reshape(*leading_shape, self.embed_dim)


class EntityTokenTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_blocks):
        super(EntityTokenTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.blocks = nn.ModuleList([
            SetAttentionBlock(embed_dim, num_heads)
            for _ in range(num_blocks)
        ])
        self.null_entity = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, entities, valid_mask):
        leading_shape = entities.shape[:-2]
        set_size = entities.shape[-2]
        entities = entities.reshape(-1, set_size, entities.shape[-1])
        valid_mask = valid_mask.reshape(-1, set_size).bool()
        encoded = self.input_projection(entities)
        empty_sets = ~valid_mask.any(dim=1, keepdim=True)
        encoded = torch.cat([
            encoded,
            self.null_entity.expand(encoded.shape[0], -1, -1)
        ], dim=1)
        safe_mask = torch.cat([valid_mask, empty_sets], dim=1)
        for block in self.blocks:
            encoded = block(encoded, safe_mask)
        return (
            encoded.reshape(*leading_shape, set_size + 1, self.embed_dim),
            safe_mask.reshape(*leading_shape, set_size + 1),
        )


class ActionEntityCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, edge_dim):
        super(ActionEntityCrossAttention, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads')
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_heads)
        )
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, queries, entities, valid_mask, edge_features,
                return_attention=False):
        batch_size, action_count, _ = queries.shape
        entity_count = entities.shape[1]
        query = self.query_projection(queries).reshape(
            batch_size, action_count, self.num_heads, self.head_dim
        )
        key = self.key_projection(entities).reshape(
            batch_size, entity_count, self.num_heads, self.head_dim
        )
        value = self.value_projection(entities).reshape(
            batch_size, entity_count, self.num_heads, self.head_dim
        )
        scores = torch.einsum('bahd,bphd->bhap', query, key)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores + self.edge_bias(edge_features).permute(0, 3, 1, 2)
        scores = scores.masked_fill(
            ~valid_mask.unsqueeze(1).unsqueeze(1),
            torch.finfo(scores.dtype).min
        )
        attention = torch.softmax(scores, dim=-1)
        context = torch.einsum('bhap,bphd->bahd', attention, value)
        context = context.reshape(batch_size, action_count, self.embed_dim)
        context = self.norm(
            queries + self.output_projection(context)
        )
        if return_attention:
            return context, attention.mean(dim=1)
        return context, None


class SharedLocalQNetwork(nn.Module):
    # Shared decentralized DQN
    def __init__(self, action_dim):
        super(SharedLocalQNetwork, self).__init__()
        if action_dim != ACTION_DIM:
            raise ValueError(f'Expected action_dim={ACTION_DIM}, got {action_dim}')
        self.action_dim = action_dim
        self.agent_id_embedding = nn.Embedding(
            NUM_AGENTS, AGENT_ID_EMBED_DIM
        )
        self.action_embedding = nn.Embedding(action_dim, ENTITY_EMBED_DIM)
        self.register_buffer('action_offsets', torch.tensor([
            [0.0, -1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0],
            [0.0, 0.0], [0.0, 0.0],
        ]) / GRID_SIZE)
        grid_center = LOCAL_GRID_RADIUS
        self.register_buffer('candidate_rows', torch.tensor([
            grid_center - 1, grid_center + 1, grid_center, grid_center,
            grid_center, grid_center,
        ]))
        self.register_buffer('candidate_cols', torch.tensor([
            grid_center, grid_center, grid_center - 1, grid_center + 1,
            grid_center, grid_center,
        ]))

        corridor_masks = torch.zeros(
            (ACTION_DIM, LOCAL_GRID_SIZE, LOCAL_GRID_SIZE),
            dtype=torch.float32,
        )
        for action, (dx, dy) in enumerate(MOVEMENT_OFFSETS):
            for distance in range(1, LOCAL_GRID_RADIUS + 1):
                center_row = grid_center + dy * distance
                center_col = grid_center + dx * distance
                for lateral in (-1, 0, 1):
                    row = center_row + (lateral if dx else 0)
                    col = center_col + (lateral if dy else 0)
                    if (0 <= row < LOCAL_GRID_SIZE
                            and 0 <= col < LOCAL_GRID_SIZE):
                        corridor_masks[action, row, col] = 1.0
        corridor_masks[ACTION_HOVER, grid_center, grid_center] = 1.0
        corridor_masks[ACTION_LAND, grid_center, grid_center] = 1.0
        self.register_buffer('corridor_masks', corridor_masks)
        self.register_buffer(
            'corridor_counts',
            corridor_masks.sum(dim=(1, 2)).clamp(min=1.0),
        )

        self.self_encoder = nn.Sequential(
            nn.Linear(DRONE_STATE_DIM + 5 + MISSION_STATE_DIM, SELF_EMBED_DIM),
            nn.ReLU(),
            nn.Linear(SELF_EMBED_DIM, SELF_EMBED_DIM),
            nn.ReLU(),
            nn.LayerNorm(SELF_EMBED_DIM)
        )
        self.patient_transformer = EntityTokenTransformer(
            13, ENTITY_EMBED_DIM, ATTENTION_HEADS, SET_ATTENTION_BLOCKS
        )
        self.drone_transformer = EntityTokenTransformer(
            24, ENTITY_EMBED_DIM, ATTENTION_HEADS, SET_ATTENTION_BLOCKS
        )
        self.patient_action_attention = ActionEntityCrossAttention(
            ENTITY_EMBED_DIM, ATTENTION_HEADS, 14
        )
        self.drone_action_attention = ActionEntityCrossAttention(
            ENTITY_EMBED_DIM, ATTENTION_HEADS, 9
        )
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                32 * LOCAL_GRID_SIZE * LOCAL_GRID_SIZE,
                GRID_EMBED_DIM,
            ),
            nn.ReLU()
        )
        base_dim = SELF_EMBED_DIM + GRID_EMBED_DIM + AGENT_ID_EMBED_DIM
        self.query_state_projection = nn.Linear(base_dim, ENTITY_EMBED_DIM)
        self.action_feature_encoder = nn.Sequential(
            nn.Linear(12, ENTITY_EMBED_DIM),
            nn.ReLU(),
            nn.Linear(ENTITY_EMBED_DIM, ENTITY_EMBED_DIM)
        )
        self.navigation_feature_encoder = nn.Sequential(
            nn.Linear(4, ENTITY_EMBED_DIM),
            nn.ReLU(),
            nn.Linear(ENTITY_EMBED_DIM, ENTITY_EMBED_DIM),
        )
        self.coordination_feature_encoder = nn.Sequential(
            nn.Linear(6, ENTITY_EMBED_DIM),
            nn.ReLU(),
            nn.Linear(ENTITY_EMBED_DIM, ENTITY_EMBED_DIM),
        )
        self.rescue_action_fusion = nn.Sequential(
            nn.Linear(3 * ENTITY_EMBED_DIM, 192),
            nn.ReLU(),
            nn.LayerNorm(192),
            nn.Linear(192, 128),
            nn.ReLU()
        )
        self.landing_action_fusion = nn.Sequential(
            nn.Linear(2 * ENTITY_EMBED_DIM, 192),
            nn.ReLU(),
            nn.LayerNorm(192),
            nn.Linear(192, 128),
            nn.ReLU()
        )
        self.rescue_value_head = nn.Sequential(
            nn.Linear(base_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.rescue_advantage_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.landing_value_head = nn.Sequential(
            nn.Linear(base_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.landing_advantage_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, observation, return_diagnostics=False):
        drones = observation['drones'].float()
        patients = observation['patients'].float()
        pending_patient_masks = observation['pending_patient_masks'].bool()
        local_grids = observation['local_grids'].float()
        mission = observation['mission'].float()
        action_masks = observation['action_masks'].bool()

        batch_size, num_agents, _ = drones.shape
        num_patients = patients.shape[1]
        drone_x = drones[:, :, 0]
        drone_y = drones[:, :, 1]
        landing_dx = drones[:, :, 5] - drone_x
        landing_dy = drones[:, :, 6] - drone_y
        landing_euclidean = torch.sqrt(
            landing_dx.square() + landing_dy.square() + 1e-8
        )
        landing_manhattan = landing_dx.abs() + landing_dy.abs()
        battery_reserve = drones[:, :, 18]
        mission_per_agent = mission.unsqueeze(1).expand(-1, num_agents, -1)
        self_features = torch.cat([
            drones,
            landing_dx.unsqueeze(-1),
            landing_dy.unsqueeze(-1),
            landing_euclidean.unsqueeze(-1),
            landing_manhattan.unsqueeze(-1),
            battery_reserve.unsqueeze(-1),
            mission_per_agent,
        ], dim=-1)
        self_embedding = self.self_encoder(self_features)

        grid_embedding = self.grid_encoder(
            local_grids.reshape(
                batch_size * num_agents,
                3,
                LOCAL_GRID_SIZE,
                LOCAL_GRID_SIZE,
            )
        ).reshape(batch_size, num_agents, GRID_EMBED_DIM)
        agent_ids = torch.arange(
            num_agents, device=drones.device
        ).unsqueeze(0).expand(batch_size, -1)
        agent_id_embedding = self.agent_id_embedding(agent_ids)
        base_embedding = torch.cat([
            self_embedding, grid_embedding, agent_id_embedding
        ], dim=-1)

        action_ids = torch.arange(
            self.action_dim, device=drones.device
        ).view(1, 1, self.action_dim).expand(
            batch_size, num_agents, -1
        )
        action_offsets = self.action_offsets.view(
            1, 1, self.action_dim, 2
        ).expand(batch_size, num_agents, -1, -1)
        candidate_grid = local_grids[
            :, :, :, self.candidate_rows, self.candidate_cols
        ].permute(0, 1, 3, 2)
        directional_grid = torch.einsum(
            'bnchw,ahw->bnac', local_grids, self.corridor_masks
        ) / self.corridor_counts.view(1, 1, self.action_dim, 1)
        landing_after_dx = landing_dx.unsqueeze(-1) - action_offsets[..., 0]
        landing_after_dy = landing_dy.unsqueeze(-1) - action_offsets[..., 1]
        landing_after_manhattan = (
            landing_after_dx.abs() + landing_after_dy.abs()
        )
        is_hover = (action_ids == ACTION_HOVER).float()
        is_land = (action_ids == ACTION_LAND).float()
        action_features = torch.cat([
            action_offsets,
            is_hover.unsqueeze(-1),
            is_land.unsqueeze(-1),
            landing_after_manhattan.unsqueeze(-1),
            action_masks.float().unsqueeze(-1),
            candidate_grid,
            directional_grid,
        ], dim=-1)
        action_queries = (
            self.query_state_projection(base_embedding).unsqueeze(2)
            + self.action_embedding(action_ids)
            + self.action_feature_encoder(action_features)
        )

        patient_x = patients[:, :, 0].unsqueeze(1).expand(-1, num_agents, -1)
        patient_y = patients[:, :, 1].unsqueeze(1).expand(-1, num_agents, -1)
        patient_dx = patient_x - drone_x.unsqueeze(-1)
        patient_dy = patient_y - drone_y.unsqueeze(-1)
        patient_euclidean = torch.sqrt(
            patient_dx.square() + patient_dy.square() + 1e-8
        )
        patient_manhattan = patient_dx.abs() + patient_dy.abs()
        timer = patients[:, :, 2].unsqueeze(1).expand(-1, num_agents, -1)
        current_weight = patients[:, :, 3].unsqueeze(1).expand(
            -1, num_agents, -1
        )
        initial_weight = patients[:, :, 4].unsqueeze(1).expand(
            -1, num_agents, -1
        )
        response_age = patients[:, :, 9].unsqueeze(1).expand(
            -1, num_agents, -1
        )
        travel_fraction = patient_manhattan * GRID_SIZE / MAX_PATIENT_TIMER
        slack = timer - travel_fraction
        reachable = (slack >= 0.0).float()
        urgency = current_weight * (1.0 + response_age)
        patient_entities = torch.stack([
            patient_dx, patient_dy, patient_euclidean, patient_manhattan,
            timer, current_weight, initial_weight, 1.0 - timer,
            response_age, travel_fraction, slack, reachable, urgency,
        ], dim=-1)
        patient_mask_per_agent = pending_patient_masks.unsqueeze(1).expand(
            -1, num_agents, num_patients
        )
        patient_tokens, safe_patient_mask = self.patient_transformer(
            patient_entities, patient_mask_per_agent
        )

        patient_dx_after = (
            patient_dx.unsqueeze(2) - action_offsets[..., 0].unsqueeze(-1)
        )
        patient_dy_after = (
            patient_dy.unsqueeze(2) - action_offsets[..., 1].unsqueeze(-1)
        )
        patient_euclidean_after = torch.sqrt(
            patient_dx_after.square() + patient_dy_after.square() + 1e-8
        )
        patient_manhattan_after = (
            patient_dx_after.abs() + patient_dy_after.abs()
        )
        slack_after = (
            timer.unsqueeze(2)
            - patient_manhattan_after * GRID_SIZE / MAX_PATIENT_TIMER
        )
        patient_to_landing = (
            (
                drones[:, :, 5].unsqueeze(-1) - patient_x
            ).abs()
            + (
                drones[:, :, 6].unsqueeze(-1) - patient_y
            ).abs()
        )
        rescue_to_pad_distance = (
            patient_manhattan_after
            + patient_to_landing.unsqueeze(2)
        )
        rescue_energy_required = (
            rescue_to_pad_distance
            * GRID_SIZE
            * BATTERY_DRAIN_PER_STEP
            * RETURN_ENERGY_RISK_MULTIPLIER
            + SAFE_RETURN_BATTERY_BUFFER
        ) / MAX_BATTERY
        rescue_energy_margin = torch.clamp(
            drones[:, :, 2].unsqueeze(-1).unsqueeze(-1)
            - rescue_energy_required,
            min=-1.0,
            max=1.0,
        )
        patient_edges = torch.stack([
            patient_dx_after,
            patient_dy_after,
            patient_euclidean_after,
            patient_manhattan_after,
            patient_manhattan.unsqueeze(2) - patient_manhattan_after,
            timer.unsqueeze(2).expand(-1, -1, self.action_dim, -1),
            current_weight.unsqueeze(2).expand(-1, -1, self.action_dim, -1),
            urgency.unsqueeze(2).expand(-1, -1, self.action_dim, -1),
            slack_after,
            (slack_after >= 0.0).float(),
            response_age.unsqueeze(2).expand(-1, -1, self.action_dim, -1),
            rescue_energy_required,
            rescue_energy_margin,
            (rescue_energy_margin >= 0.0).float(),
        ], dim=-1)
        patient_edges = torch.cat([
            patient_edges,
            torch.zeros_like(patient_edges[..., :1, :])
        ], dim=-2)

        pending_per_agent = pending_patient_masks.unsqueeze(1).expand(
            -1, num_agents, -1
        )
        has_pending = pending_per_agent.any(dim=-1)
        masked_patient_distance = patient_manhattan.masked_fill(
            ~pending_per_agent, 2.0
        )
        nearest_patient_distance = masked_patient_distance.min(dim=-1).values
        pending_per_action = pending_per_agent.unsqueeze(2).expand(
            -1, -1, self.action_dim, -1
        )
        nearest_patient_after = patient_manhattan_after.masked_fill(
            ~pending_per_action, 2.0
        ).min(dim=-1).values
        nearest_patient_distance = torch.where(
            has_pending,
            nearest_patient_distance,
            torch.zeros_like(nearest_patient_distance),
        )
        nearest_patient_after = torch.where(
            has_pending.unsqueeze(-1),
            nearest_patient_after,
            torch.zeros_like(nearest_patient_after),
        )
        nearest_patient_progress_cells = (
            nearest_patient_distance.unsqueeze(-1) - nearest_patient_after
        ) * GRID_SIZE
        landing_progress_cells = (
            landing_manhattan.unsqueeze(-1) - landing_after_manhattan
        ) * GRID_SIZE
        navigation_features = torch.stack([
            nearest_patient_after,
            nearest_patient_progress_cells,
            landing_after_manhattan,
            landing_progress_cells,
        ], dim=-1)
        action_queries = (
            action_queries
            + self.navigation_feature_encoder(navigation_features)
        )

        other_x = drone_x.unsqueeze(1).expand(-1, num_agents, -1)
        other_y = drone_y.unsqueeze(1).expand(-1, num_agents, -1)
        other_dx = other_x - drone_x.unsqueeze(2)
        other_dy = other_y - drone_y.unsqueeze(2)
        other_distance = torch.sqrt(
            other_dx.square() + other_dy.square() + 1e-8
        )
        normalized_closeness = CLOSENESS_RADIUS / GRID_SIZE
        other_indices = torch.arange(
            num_agents, device=drones.device
        ).view(1, 1, num_agents).expand(batch_size, num_agents, -1)
        ego_indices = torch.arange(
            num_agents, device=drones.device
        ).view(1, num_agents, 1).expand(batch_size, -1, num_agents)
        other_role = other_indices.float() / max(1, num_agents - 1)
        role_delta = (
            (other_indices - ego_indices).float() / max(1, num_agents - 1)
        )
        other_previous_actions = drones[:, :, 11:11 + ACTION_DIM].unsqueeze(1).expand(
            -1, num_agents, -1, -1
        )
        other_cross_layer = drones[:, :, 17:22].unsqueeze(1).expand(
            -1, num_agents, -1, -1
        )
        other_entities = torch.cat([
            torch.stack([
                other_dx,
                other_dy,
                other_distance,
                drones[:, :, 2].unsqueeze(1).expand(-1, num_agents, -1),
                drones[:, :, 3].unsqueeze(1).expand(-1, num_agents, -1),
                drones[:, :, 4].unsqueeze(1).expand(-1, num_agents, -1),
                drones[:, :, 7].unsqueeze(1).expand(-1, num_agents, -1),
                drones[:, :, 8].unsqueeze(1).expand(-1, num_agents, -1),
                drones[:, :, 9].unsqueeze(1).expand(-1, num_agents, -1),
                drones[:, :, 10].unsqueeze(1).expand(-1, num_agents, -1),
                (other_distance <= normalized_closeness).float(),
                other_role,
                role_delta,
            ], dim=-1),
            other_previous_actions,
            other_cross_layer,
        ], dim=-1)
        self_mask = torch.eye(
            num_agents, device=drones.device, dtype=torch.bool
        ).unsqueeze(0)
        other_mask = (~self_mask).expand(batch_size, -1, -1)
        drone_tokens, safe_drone_mask = self.drone_transformer(
            other_entities, other_mask
        )

        candidate_x = drone_x.unsqueeze(-1) + action_offsets[..., 0]
        candidate_y = drone_y.unsqueeze(-1) + action_offsets[..., 1]
        other_dx_after = (
            other_x.unsqueeze(2) - candidate_x.unsqueeze(-1)
        )
        other_dy_after = (
            other_y.unsqueeze(2) - candidate_y.unsqueeze(-1)
        )
        other_distance_after = torch.sqrt(
            other_dx_after.square() + other_dy_after.square() + 1e-8
        )
        other_manhattan_after = other_dx_after.abs() + other_dy_after.abs()
        predicted_other_x = (
            other_x + drones[:, :, 7].unsqueeze(1).expand(-1, num_agents, -1)
            / GRID_SIZE
        )
        predicted_other_y = (
            other_y + drones[:, :, 8].unsqueeze(1).expand(-1, num_agents, -1)
            / GRID_SIZE
        )
        predicted_dx = predicted_other_x.unsqueeze(2) - candidate_x.unsqueeze(-1)
        predicted_dy = predicted_other_y.unsqueeze(2) - candidate_y.unsqueeze(-1)
        predicted_distance = torch.sqrt(
            predicted_dx.square() + predicted_dy.square() + 1e-8
        )
        same_predicted_destination = (
            (predicted_dx.abs() < 0.5 / GRID_SIZE)
            & (predicted_dy.abs() < 0.5 / GRID_SIZE)
        ).float()
        head_on_prediction = (
            (other_dx_after.abs() < 0.5 / GRID_SIZE)
            & (other_dy_after.abs() < 0.5 / GRID_SIZE)
            & ((predicted_other_x.unsqueeze(2) - drone_x.unsqueeze(-1).unsqueeze(-1)).abs()
               < 0.5 / GRID_SIZE)
            & ((predicted_other_y.unsqueeze(2) - drone_y.unsqueeze(-1).unsqueeze(-1)).abs()
               < 0.5 / GRID_SIZE)
        ).float()
        active_other_mask = (
            other_mask
            & ~drones[:, :, 3].bool().unsqueeze(1)
            & ~drones[:, :, 4].bool().unsqueeze(1)
        )
        active_other_per_action = active_other_mask.unsqueeze(2).expand(
            -1, -1, self.action_dim, -1
        )
        active_other_count = active_other_per_action.float().sum(
            dim=-1
        ).clamp(min=1.0)
        has_active_other = active_other_per_action.any(dim=-1)
        minimum_other_distance_after = other_distance_after.masked_fill(
            ~active_other_per_action, 2.0
        ).min(dim=-1).values
        maximum_approach_cells = (
            other_distance.unsqueeze(2) - other_distance_after
        ).masked_fill(~active_other_per_action, -2.0).max(dim=-1).values * GRID_SIZE
        minimum_other_distance_after = torch.where(
            has_active_other,
            minimum_other_distance_after,
            torch.zeros_like(minimum_other_distance_after),
        )
        maximum_approach_cells = torch.where(
            has_active_other,
            maximum_approach_cells,
            torch.zeros_like(maximum_approach_cells),
        )
        candidate_occupied = (
            (other_dx_after.abs() < 0.5 / GRID_SIZE)
            & (other_dy_after.abs() < 0.5 / GRID_SIZE)
        ).float()
        ego_yields_to_other = (
            ego_indices > other_indices
        ).float().unsqueeze(2).expand(-1, -1, self.action_dim, -1)
        coordination_features = torch.stack([
            minimum_other_distance_after,
            maximum_approach_cells,
            (candidate_occupied * active_other_per_action.float()).sum(dim=-1)
            / active_other_count,
            (same_predicted_destination * active_other_per_action.float()).sum(
                dim=-1
            ) / active_other_count,
            (head_on_prediction * active_other_per_action.float()).sum(dim=-1)
            / active_other_count,
            (
                same_predicted_destination * ego_yields_to_other
                * active_other_per_action.float()
            ).sum(dim=-1) / active_other_count,
        ], dim=-1)
        action_queries = (
            action_queries
            + self.coordination_feature_encoder(coordination_features)
        )
        drone_edges = torch.stack([
            other_dx_after,
            other_dy_after,
            other_distance_after,
            other_manhattan_after,
            other_distance.unsqueeze(2) - other_distance_after,
            predicted_distance,
            same_predicted_destination,
            head_on_prediction,
            role_delta.unsqueeze(2).expand(-1, -1, self.action_dim, -1),
        ], dim=-1)
        drone_edges = torch.cat([
            drone_edges,
            torch.zeros_like(drone_edges[..., :1, :])
        ], dim=-2)

        flat_count = batch_size * num_agents
        flat_queries = action_queries.reshape(
            flat_count, self.action_dim, ENTITY_EMBED_DIM
        )
        patient_context, patient_attention = self.patient_action_attention(
            flat_queries,
            patient_tokens.reshape(flat_count, num_patients + 1, ENTITY_EMBED_DIM),
            safe_patient_mask.reshape(flat_count, num_patients + 1),
            patient_edges.reshape(
                flat_count, self.action_dim, num_patients + 1, 14
            ),
            return_attention=return_diagnostics
        )
        drone_context, drone_attention = self.drone_action_attention(
            flat_queries,
            drone_tokens.reshape(flat_count, num_agents + 1, ENTITY_EMBED_DIM),
            safe_drone_mask.reshape(flat_count, num_agents + 1),
            drone_edges.reshape(
                flat_count, self.action_dim, num_agents + 1, 9
            ),
            return_attention=return_diagnostics
        )
        fp32_head_context = (
            torch.autocast(device_type='cuda', enabled=False)
            if drones.device.type == 'cuda' else nullcontext()
        )
        with fp32_head_context:
            rescue_action_latent = self.rescue_action_fusion(torch.cat([
                flat_queries, patient_context, drone_context
            ], dim=-1).float()).reshape(
                batch_size, num_agents, self.action_dim, 128
            )
            landing_action_latent = self.landing_action_fusion(torch.cat([
                flat_queries, drone_context
            ], dim=-1).float()).reshape(
                batch_size, num_agents, self.action_dim, 128
            )

            valid_action_count = action_masks.sum(
                dim=-1, keepdim=True
            ).clamp(min=1).float()
            float_action_masks = action_masks.float()

            def phase_q_values(value_head, advantage_head, phase_latent):
                advantage = advantage_head(phase_latent).squeeze(-1)
                valid_advantage_mean = (
                    (advantage * float_action_masks).sum(
                        dim=-1, keepdim=True
                    ) / valid_action_count
                )
                return (
                    value_head(base_embedding.float())
                    + advantage - valid_advantage_mean
                )

            rescue_q_values = phase_q_values(
                self.rescue_value_head,
                self.rescue_advantage_head,
                rescue_action_latent,
            )
            landing_q_values = phase_q_values(
                self.landing_value_head,
                self.landing_advantage_head,
                landing_action_latent,
            )
            global_landing_phase = mission[:, 8].reshape(
                batch_size, 1
            ).float()
            agent_return_phase = drones[:, :, -1].float()
            landing_phase = torch.maximum(
                global_landing_phase, agent_return_phase
            ).unsqueeze(-1)
            q_values = (
                (1.0 - landing_phase) * rescue_q_values
                + landing_phase * landing_q_values
            )
            action_latent = (
                (1.0 - landing_phase.unsqueeze(-1)) * rescue_action_latent
                + landing_phase.unsqueeze(-1) * landing_action_latent
            )

        if not return_diagnostics:
            return q_values
        normalized_queries = torch.nn.functional.normalize(
            action_latent, dim=-1
        )
        query_similarity = torch.matmul(
            normalized_queries, normalized_queries.transpose(-1, -2)
        )
        off_diagonal = ~torch.eye(
            self.action_dim, device=drones.device, dtype=torch.bool
        )
        diagnostics = {
            'patient_attention': patient_attention.reshape(
                batch_size, num_agents, self.action_dim, num_patients + 1
            ),
            'drone_attention': drone_attention.reshape(
                batch_size, num_agents, self.action_dim, num_agents + 1
            ),
            'patient_attention_mask': safe_patient_mask,
            'drone_attention_mask': safe_drone_mask,
            'action_query_similarity': query_similarity[..., off_diagonal].mean(),
            'action_latent_norm': action_latent.norm(dim=-1).mean(),
            'rescue_q_values': rescue_q_values,
            'landing_q_values': landing_q_values,
            'q_value_dtype': str(q_values.dtype),
        }
        return q_values, diagnostics


# Centralized QMIX networks
class CentralStateEncoder(nn.Module):
    def __init__(self):
        super(CentralStateEncoder, self).__init__()
        self.agent_id_embedding = nn.Embedding(
            NUM_AGENTS, AGENT_ID_EMBED_DIM
        )
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                8 * LOCAL_GRID_SIZE * LOCAL_GRID_SIZE,
                CENTRAL_GRID_EMBED_DIM,
            ),
            nn.ReLU()
        )
        agent_entity_dim = (
            DRONE_STATE_DIM + CENTRAL_GRID_EMBED_DIM + AGENT_ID_EMBED_DIM
        )
        self.drone_transformer = EntitySetTransformer(
            agent_entity_dim,
            ENTITY_EMBED_DIM,
            ATTENTION_HEADS,
            SET_ATTENTION_BLOCKS
        )
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_entity_dim, ENTITY_EMBED_DIM),
            nn.ReLU(),
            nn.LayerNorm(ENTITY_EMBED_DIM)
        )
        self.patient_transformer = EntitySetTransformer(
            PATIENT_STATE_DIM,
            ENTITY_EMBED_DIM,
            ATTENTION_HEADS,
            SET_ATTENTION_BLOCKS
        )
        self.mission_encoder = nn.Sequential(
            nn.Linear(MISSION_STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(2 * ENTITY_EMBED_DIM + 64, GLOBAL_EMBED_DIM),
            nn.ReLU(),
            nn.LayerNorm(GLOBAL_EMBED_DIM)
        )
        self.agent_output = nn.Sequential(
            nn.Linear(
                ENTITY_EMBED_DIM + GLOBAL_EMBED_DIM,
                AGENT_MIX_CONTEXT_DIM
            ),
            nn.ReLU(),
            nn.LayerNorm(AGENT_MIX_CONTEXT_DIM)
        )

    def forward(self, observation):
        drones = observation['drones'].float()
        patients = observation['patients'].float()
        patient_masks = observation['patient_masks'].bool()
        local_grids = observation['local_grids'].float()
        mission = observation['mission'].float()

        batch_size, num_agents, _ = drones.shape
        grid_embedding = self.grid_encoder(
            local_grids.reshape(
                batch_size * num_agents,
                3,
                LOCAL_GRID_SIZE,
                LOCAL_GRID_SIZE,
            )
        ).reshape(batch_size, num_agents, CENTRAL_GRID_EMBED_DIM)
        agent_ids = torch.arange(
            num_agents, device=drones.device
        ).unsqueeze(0).expand(batch_size, -1)
        agent_id_embedding = self.agent_id_embedding(agent_ids)
        agent_entities = torch.cat([
            drones, grid_embedding, agent_id_embedding
        ], dim=-1)

        drone_mask = torch.ones(
            (batch_size, num_agents), device=drones.device, dtype=torch.bool
        )
        drone_context = self.drone_transformer(agent_entities, drone_mask)
        individual_agent_context = self.agent_encoder(agent_entities)
        patient_context = self.patient_transformer(patients, patient_masks)
        mission_context = self.mission_encoder(mission)
        global_context = self.output(torch.cat([
            drone_context,
            patient_context,
            mission_context,
        ], dim=-1))
        agent_context = self.agent_output(torch.cat([
            individual_agent_context,
            global_context.unsqueeze(1).expand(-1, num_agents, -1),
        ], dim=-1))
        return global_context, agent_context


class QMixer(nn.Module):
    def __init__(self):
        super(QMixer, self).__init__()
        self.state_encoder = CentralStateEncoder()
        self.hyper_w1 = nn.Sequential(
            nn.Linear(AGENT_MIX_CONTEXT_DIM, AGENT_MIX_CONTEXT_DIM),
            nn.ReLU(),
            nn.Linear(AGENT_MIX_CONTEXT_DIM, MIXER_EMBED_DIM)
        )
        self.hyper_b1 = nn.Linear(GLOBAL_EMBED_DIM, MIXER_EMBED_DIM)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(GLOBAL_EMBED_DIM, GLOBAL_EMBED_DIM),
            nn.ReLU(),
            nn.Linear(GLOBAL_EMBED_DIM, MIXER_EMBED_DIM)
        )
        self.state_value = nn.Sequential(
            nn.Linear(GLOBAL_EMBED_DIM, MIXER_EMBED_DIM),
            nn.ReLU(),
            nn.Linear(MIXER_EMBED_DIM, 1)
        )

    def forward(self, agent_utilities, observation, agent_mask,
                return_diagnostics=False):
        batch_size = agent_utilities.shape[0]
        global_embedding, agent_context = self.state_encoder(observation)
        utilities = (agent_utilities * agent_mask).view(
            batch_size, 1, NUM_AGENTS
        )

        active_count = agent_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        weights1 = (
            MIXER_MIN_RAW_WEIGHT
            + (2.0 - MIXER_MIN_RAW_WEIGHT)
            * torch.sigmoid(self.hyper_w1(agent_context))
        ) / active_count.unsqueeze(-1)
        bias1 = self.hyper_b1(global_embedding).view(
            batch_size, 1, MIXER_EMBED_DIM
        )
        pre_hidden = torch.bmm(utilities, weights1) + bias1
        hidden = torch.nn.functional.elu(pre_hidden)
        weights2 = (
            MIXER_MIN_RAW_WEIGHT
            + (2.0 - MIXER_MIN_RAW_WEIGHT)
            * torch.sigmoid(self.hyper_w2(global_embedding))
        ) / MIXER_EMBED_DIM
        weights2 = weights2.view(batch_size, MIXER_EMBED_DIM, 1)
        state_value = self.state_value(global_embedding).view(batch_size, 1, 1)
        mixed_utility = torch.bmm(hidden, weights2)
        q_total = (mixed_utility + state_value).view(batch_size, 1)
        if not return_diagnostics:
            return q_total

        elu_derivative = torch.where(
            pre_hidden > 0.0, torch.ones_like(pre_hidden), torch.exp(pre_hidden)
        )
        effective_sensitivity = torch.bmm(
            weights1,
            elu_derivative.transpose(1, 2) * weights2
        ).squeeze(-1)
        return q_total, {
            'state_value': state_value.view(batch_size, 1),
            'mixed_utility': mixed_utility.view(batch_size, 1),
            'mixer_weight_mean': torch.cat([
                weights1.reshape(batch_size, -1),
                weights2.reshape(batch_size, -1),
            ], dim=1).mean(dim=1),
            'mixer_weight_min': torch.minimum(
                weights1.amin(dim=(1, 2)), weights2.amin(dim=(1, 2))
            ),
            'mixer_weight_max': torch.maximum(
                weights1.amax(dim=(1, 2)), weights2.amax(dim=(1, 2))
            ),
            'utility_sensitivity': effective_sensitivity,
        }


# Checkpoint loading and decentralized inference
class CEDAFGCSPX4Policy:
    REQUIRED_SHAPES = {
        "drones": (NUM_AGENTS, DRONE_STATE_DIM),
        "patients": (MAX_PATIENTS, PATIENT_STATE_DIM),
        "patient_masks": (MAX_PATIENTS,),
        "pending_patient_masks": (MAX_PATIENTS,),
        "local_grids": (
            NUM_AGENTS, 3, LOCAL_GRID_SIZE, LOCAL_GRID_SIZE
        ),
        "mission": (MISSION_STATE_DIM,),
        "action_masks": (NUM_AGENTS, ACTION_DIM),
    }
    BOOLEAN_FIELDS = {
        "patient_masks", "pending_patient_masks", "action_masks"
    }

    def __init__(self, weights_path=None, device="auto", load_mixer=False):
        self.device = resolve_device(device)
        if weights_path is None:
            weights_path = (
                Path(__file__).resolve().parent
                / "weights"
                / CHECKPOINT_FILENAME
            )
        self.weights_path = Path(weights_path).expanduser().resolve()
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")

        try:
            checkpoint = torch.load(
                self.weights_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                self.weights_path, map_location="cpu"
            )
        self._validate_checkpoint(checkpoint)

        self.policy_net = SharedLocalQNetwork(ACTION_DIM).to(self.device)
        self.policy_net.load_state_dict(
            checkpoint["policy_state_dict"], strict=True
        )
        self.policy_net.eval()

        self.mixer = None
        if load_mixer:
            self.mixer = QMixer().to(self.device)
            self.mixer.load_state_dict(
                checkpoint["mixer_state_dict"], strict=True
            )
            self.mixer.eval()

        self.metadata = {
            key: checkpoint.get(key)
            for key in (
                "model_version", "global_step", "learner_steps",
                "num_agents", "grid_size", "max_patients", "action_dim",
                "action_names", "drone_state_dim", "patient_state_dim",
                "mission_state_dim", "local_grid_radius", "local_grid_size",
                "maximum_battery", "battery_drain_per_step",
                "battery_drain_in_wind", "clean_endurance_steps",
                "gamma", "n_step",
            )
        }
        del checkpoint

    @staticmethod
    def _validate_checkpoint(checkpoint):
        expected = {
            "model_version": MODEL_VERSION,
            "num_agents": NUM_AGENTS,
            "grid_size": GRID_SIZE,
            "max_patients": MAX_PATIENTS,
            "action_dim": ACTION_DIM,
            "action_names": ACTION_NAMES,
            "drone_state_dim": DRONE_STATE_DIM,
            "patient_state_dim": PATIENT_STATE_DIM,
            "mission_state_dim": MISSION_STATE_DIM,
            "local_grid_radius": LOCAL_GRID_RADIUS,
            "local_grid_size": LOCAL_GRID_SIZE,
            "maximum_battery": MAX_BATTERY,
            "battery_drain_per_step": BATTERY_DRAIN_PER_STEP,
            "battery_drain_in_wind": BATTERY_DRAIN_IN_WIND,
        }
        for key, value in expected.items():
            actual = checkpoint.get(key)
            if key == "action_names" and actual is not None:
                actual = tuple(actual)
            if actual != value:
                raise ValueError(
                    f"Checkpoint {key} mismatch: "
                    f"{actual!r} != {value!r}"
                )
        for key in ("policy_state_dict", "mixer_state_dict"):
            if key not in checkpoint:
                raise KeyError(f"Checkpoint is missing {key}")

    def prepare_observation(self, observation: Mapping[str, object]):
        missing = set(self.REQUIRED_SHAPES) - set(observation)
        if missing:
            raise KeyError(f"Observation is missing: {sorted(missing)}")

        prepared = {}
        for key, shape in self.REQUIRED_SHAPES.items():
            value = np.asarray(observation[key])
            if value.shape != shape:
                raise ValueError(
                    f"{key} must have shape {shape}, got {value.shape}"
                )
            dtype = (
                torch.bool if key in self.BOOLEAN_FIELDS
                else torch.float32
            )
            if dtype == torch.float32 and not np.isfinite(value).all():
                raise ValueError(f"{key} contains a non-finite value")
            prepared[key] = torch.as_tensor(
                value, dtype=dtype, device=self.device
            ).unsqueeze(0)

        if bool((
                prepared["pending_patient_masks"]
                & ~prepared["patient_masks"]
        ).any()):
            raise ValueError(
                "pending_patient_masks must be a subset of patient_masks"
            )
        valid_actions = prepared["action_masks"].any(dim=-1)
        if not bool(valid_actions.all()):
            raise ValueError("Every drone must have at least one valid action")
        return prepared

    def q_values(self, observation, mask_invalid=False):
        prepared = self.prepare_observation(observation)
        with torch.inference_mode():
            values = self.policy_net(prepared).float()
            if mask_invalid:
                values = values.masked_fill(
                    ~prepared["action_masks"], -torch.inf
                )
        return values[0].cpu().numpy()

    def select_actions(self, observation):
        prepared = self.prepare_observation(observation)
        with torch.inference_mode():
            values = self.policy_net(prepared).float()
            values = values.masked_fill(
                ~prepared["action_masks"], -torch.inf
            )
            actions = values.argmax(dim=-1)[0]
        return actions.cpu().tolist()

    def select_agent_action(self, observation, agent_index):
        if not 0 <= int(agent_index) < NUM_AGENTS:
            raise IndexError(
                f"agent_index must be from 0 through {NUM_AGENTS - 1}"
            )
        return self.select_actions(observation)[int(agent_index)]

    def mix_selected_utilities(
            self, observation, actions: Sequence[int], agent_mask=None):
        if self.mixer is None:
            raise RuntimeError(
                "The training-only mixer was not loaded; construct the policy "
                "with load_mixer=True for offline diagnostics"
            )
        prepared = self.prepare_observation(observation)
        if len(actions) != NUM_AGENTS:
            raise ValueError(f"Expected {NUM_AGENTS} actions")
        if any(int(action) not in range(ACTION_DIM) for action in actions):
            raise ValueError(f"Actions must be in [0, {ACTION_DIM - 1}]")
        actions = torch.as_tensor(
            actions, dtype=torch.long, device=self.device
        ).reshape(1, NUM_AGENTS, 1)
        if agent_mask is None:
            agent_mask = torch.ones(
                (1, NUM_AGENTS), dtype=torch.float32, device=self.device
            )
        else:
            if len(agent_mask) != NUM_AGENTS:
                raise ValueError(f"Expected {NUM_AGENTS} agent-mask values")
            agent_mask = torch.as_tensor(
                agent_mask, dtype=torch.float32, device=self.device
            ).reshape(1, NUM_AGENTS)
        with torch.inference_mode():
            q_values = self.policy_net(prepared).float()
            utilities = q_values.gather(dim=-1, index=actions).squeeze(-1)
            q_total = self.mixer(utilities, prepared, agent_mask)
        return float(q_total.item())


# Command-line package verification
def resolve_device(device_name):
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def make_smoke_test_observation():
    observation = {
        "drones": np.zeros(
            (NUM_AGENTS, DRONE_STATE_DIM), dtype=np.float32
        ),
        "patients": np.zeros(
            (MAX_PATIENTS, PATIENT_STATE_DIM), dtype=np.float32
        ),
        "patient_masks": np.zeros(MAX_PATIENTS, dtype=bool),
        "pending_patient_masks": np.zeros(MAX_PATIENTS, dtype=bool),
        "local_grids": np.zeros(
            (NUM_AGENTS, 3, LOCAL_GRID_SIZE, LOCAL_GRID_SIZE),
            dtype=np.float32,
        ),
        "mission": np.zeros(MISSION_STATE_DIM, dtype=np.float32),
        "action_masks": np.zeros(
            (NUM_AGENTS, ACTION_DIM), dtype=bool
        ),
    }
    observation["drones"][:, 2] = 1.0
    observation["action_masks"][:, ACTION_HOVER] = True
    return observation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify or inspect the CEDA-FGCS-PX4 deployment model."
    )
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--mixer-diagnostic",
        action="store_true",
        help="Also load and run the training-only QMIX mixer.",
    )
    parser.add_argument("--show-metadata", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model = CEDAFGCSPX4Policy(
        args.weights,
        args.device,
        load_mixer=args.mixer_diagnostic,
    )
    if args.show_metadata:
        for key, value in model.metadata.items():
            print(f"{key}: {value}")
    if args.smoke_test:
        observation = make_smoke_test_observation()
        actions = model.select_actions(observation)
        action_names = [ACTION_NAMES[action] for action in actions]
        print(f"Strict weight load: OK ({model.weights_path})")
        print(f"Device: {model.device}")
        print(f"Actions: {actions} -> {action_names}")
        if args.mixer_diagnostic:
            q_total = model.mix_selected_utilities(observation, actions)
            print(f"QMIX diagnostic value: {q_total:.6f}")
    if not args.smoke_test and not args.show_metadata:
        print("Model loaded. Use --smoke-test or --show-metadata.")


if __name__ == "__main__":
    main()
