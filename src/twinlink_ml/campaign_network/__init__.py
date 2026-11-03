"""Campaign network: does a heat network pay when the consumers do not run all the time.

One constant waste-heat source (compressor station, water as carrier) and consumers with on/off
campaigns (process-air preheaters of plants A, B, C) plus low-temperature sinks (hall heating,
process water). Configurations: K0 no network (source fixed to plant A), K1 network with the
cascade rule, K2 K1 plus a storage rule, K3 linear programme with full foresight (upper bound).
Study document ``docs/35_campaign_network.md``; settings
``conf/campaign_network/campaign_network.yaml``.
"""

from .cascade import CascadeResult, cascade_stages, run_k0, run_k1, run_k2, stage_duty
from .config import (
    DEFAULT_CONFIG_PATH,
    CampaignNetworkConfig,
    load_campaign_network_config,
    variant_names,
)
from .evaluate import evaluate_draw, key_figures
from .profiles import ambient_temperature, build_consumers, campaign_schedule, plant_schedules
from .target import TargetResult, run_k3

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "CampaignNetworkConfig",
    "CascadeResult",
    "TargetResult",
    "ambient_temperature",
    "build_consumers",
    "campaign_schedule",
    "cascade_stages",
    "evaluate_draw",
    "key_figures",
    "load_campaign_network_config",
    "plant_schedules",
    "run_k0",
    "run_k1",
    "run_k2",
    "run_k3",
    "stage_duty",
    "variant_names",
]
