"""Procedural generator for Phase 2 long-session evaluation data.

NAVIGATOR_CONTINUATION_SPEC §Dataset Generation.

Produces three variants of a 50-interaction + 10-query session:
  - 20 supply-chain interactions (domain A)
  - 20 hospital-operations interactions (domain B)
  - 10 cybersecurity interactions     (domain C)
  - 10 cross-domain queries (3 recall + 3 analogy + 2 topology + 2 scope)

Each variant uses different entity names but identical structural patterns
so the navigator's pattern library behaviour can be compared across runs.

Output: one JSONL per variant plus a manifest JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


# ----------------------------------------------------------------------
# Entity pools — three variants. All three define the same roles but use
# different surface names so the navigator's signature matching can be
# tested on structurally-equivalent graphs.
# ----------------------------------------------------------------------


ENTITY_POOLS: List[Dict[str, str]] = [
    {   # Variant 0
        # Supply chain
        "port_asia": "shanghai_port",
        "port_eu": "rotterdam_port",
        "port_us": "la_hub",
        "warehouse_us": "phoenix_warehouse",
        "route_eu": "hamburg_munich",
        "route_asia": "shanghai_shenzhen",
        "fab_origin": "taiwan_fab",
        "disruptor_asia": "taiwan_typhoon",
        "city_eu": "munich",
        "retailer": "us_specialty_retailer",
        "disruptor_agri": "midwest_drought",
        "commodity_agri": "wheat",
        "consumer_product_a": "bread",
        "consumer_product_b": "pasta",
        # Hospital
        "hospital": "city_general",
        "er": "emergency_dept",
        "icu": "icu_wing",
        "dept_rad": "radiology",
        "dept_surg": "surgery",
        "equipment_key": "mri_scanner",
        "supply_consumable": "surgical_gauze",
        "role_junior": "junior_nurse",
        "role_senior": "attending_physician",
        "action_controlled": "controlled_substance_order",
        "action_routine": "routine_med_order",
        # Cybersecurity
        "entry_point": "unpatched_vpn",
        "lateral_target": "domain_controller",
        "terminal_system": "billing_system",
        "critical_service": "patient_intake",
        "attacker": "ransomware",
        "role_analyst_junior": "tier1_analyst",
        "role_analyst_senior": "senior_architect",
        "action_cyber_controlled": "prod_firewall_change",
    },
    {   # Variant 1
        "port_asia": "singapore_port",
        "port_eu": "antwerp_port",
        "port_us": "long_beach_hub",
        "warehouse_us": "atlanta_warehouse",
        "route_eu": "zurich_milan",
        "route_asia": "singapore_bangkok",
        "fab_origin": "korea_fab",
        "disruptor_asia": "korea_earthquake",
        "city_eu": "zurich",
        "retailer": "east_coast_retailer",
        "disruptor_agri": "plains_frost",
        "commodity_agri": "corn",
        "consumer_product_a": "tortillas",
        "consumer_product_b": "cornflakes",
        "hospital": "metro_medical",
        "er": "trauma_center",
        "icu": "ccu_wing",
        "dept_rad": "imaging",
        "dept_surg": "operating_room",
        "equipment_key": "ct_scanner",
        "supply_consumable": "sutures",
        "role_junior": "resident_physician",
        "role_senior": "chief_surgeon",
        "action_controlled": "narcotic_prescription",
        "action_routine": "lab_order",
        "entry_point": "exposed_rdp",
        "lateral_target": "ad_forest",
        "terminal_system": "erp_system",
        "critical_service": "supply_reorder",
        "attacker": "apt_group",
        "role_analyst_junior": "soc_analyst",
        "role_analyst_senior": "incident_commander",
        "action_cyber_controlled": "kernel_deploy",
    },
    {   # Variant 2
        "port_asia": "busan_port",
        "port_eu": "felixstowe_port",
        "port_us": "newark_hub",
        "warehouse_us": "dallas_warehouse",
        "route_eu": "vienna_prague",
        "route_asia": "manila_tokyo",
        "fab_origin": "philippines_fab",
        "disruptor_asia": "philippines_flood",
        "city_eu": "vienna",
        "retailer": "midwest_retailer",
        "disruptor_agri": "california_heatwave",
        "commodity_agri": "tomato",
        "consumer_product_a": "ketchup",
        "consumer_product_b": "pasta_sauce",
        "hospital": "st_anselm",
        "er": "acute_care_dept",
        "icu": "pediatric_icu",
        "dept_rad": "nuclear_imaging",
        "dept_surg": "cardiac_surgery",
        "equipment_key": "linac",
        "supply_consumable": "saline_bags",
        "role_junior": "nurse_extern",
        "role_senior": "medical_director",
        "action_controlled": "chemo_administration",
        "action_routine": "vital_check",
        "entry_point": "phishing_email",
        "lateral_target": "file_server",
        "terminal_system": "claims_processor",
        "critical_service": "insurance_auth",
        "attacker": "ransomware_gang",
        "role_analyst_junior": "tier2_analyst",
        "role_analyst_senior": "ciso_delegate",
        "action_cyber_controlled": "prod_db_patch",
    },
]


# ----------------------------------------------------------------------
# Structural templates — one per interaction slot. Each template returns
# (text, fragment) given the variant's entity pool.
# ----------------------------------------------------------------------


def _sub(e: Dict[str, str], s: str) -> str:
    """Substitute {{key}} tokens in `s` with entities from pool `e`."""
    for k, v in e.items():
        s = s.replace("{{" + k + "}}", v)
    return s


# ----------------------------------------------------------------------
# Stylization — wrap the structural claim in realistic operational prose.
# Every interaction runs through one of these styles so the text the
# LLM reads looks like something from an actual enterprise comms system
# (Slack, email, post-mortem, ticketing, exec brief). The structural
# content is preserved verbatim; only the prose frame changes.
# ----------------------------------------------------------------------


import hashlib


_MONTHS = ["Jan 14", "Jan 21", "Feb 3", "Feb 17", "Mar 4", "Mar 11",
           "Mar 25", "Apr 2", "Apr 18", "May 6", "May 19", "Jun 7",
           "Jun 24", "Jul 9", "Jul 22", "Aug 5", "Aug 19", "Sep 3",
           "Sep 16", "Oct 1"]


def _frame(turn: int, domain: str) -> str:
    """Deterministic prefix per (turn, domain). No shuffling — re-runs
    produce identical prose for the same turn. Not random, keyed on
    turn so variance across turns is high but reproducible."""
    day = _MONTHS[turn % len(_MONTHS)]
    if domain == "supply_chain":
        options = [
            f"[Ops standup, {day}]",
            f"[Slack #logistics, {day}]",
            f"[Exec brief, {day}]",
            f"[Supplier risk memo, {day}]",
            f"[Trading desk note, {day}]",
        ]
    elif domain == "hospital":
        options = [
            f"[CNO daily rollup, {day}]",
            f"[Incident report, {day}]",
            f"[Clinical policy memo, {day}]",
            f"[Pharmacy bulletin, {day}]",
            f"[Board briefing, {day}]",
        ]
    elif domain == "cybersecurity":
        options = [
            f"[SOC ticket, {day}]",
            f"[Security policy memo, {day}]",
            f"[Incident post-mortem, {day}]",
            f"[CISO advisory, {day}]",
            f"[Change mgmt memo, {day}]",
        ]
    else:
        options = [f"[{day}]"]
    return options[turn % len(options)]


def _salt(turn: int, domain: str) -> str:
    """Plausible tangential detail that doesn't change the graph.
    Keyed by turn so results are reproducible."""
    h = int(hashlib.sha1(f"{turn}:{domain}".encode()).hexdigest(), 16)
    pool = {
        "supply_chain": [
            "Logistics lead flagged this on the 9am call.",
            "Forwarders are already re-quoting on spot.",
            "Finance will update the quarterly guidance.",
            "Procurement is exploring backup sourcing.",
            "The risk committee reviewed last Tuesday.",
            "Board pack for this Friday will include it.",
            "WoW throughput down ~15%, trending worse.",
            "Three shipments under claim review.",
            "Comms drafted for investor Q&A.",
            "Expected to persist through end of month.",
        ],
        "hospital": [
            "Nursing leadership briefed this morning.",
            "Patient safety officer was copied.",
            "The on-call attending raised it at handoff.",
            "Compliance requested documentation by Friday.",
            "Clinical director asked for a root-cause memo.",
            "Budget impact ~$120K this fiscal.",
            "Incident logged in the quality dashboard.",
            "Risk management flagged regulatory exposure.",
            "Patient complaints increased 22% QoQ.",
            "Licensure implications under review.",
        ],
        "cybersecurity": [
            "Tier-2 escalation opened in the SIEM.",
            "Threat intel confirms known-actor TTPs.",
            "Legal was notified per the IR playbook.",
            "Business impact estimated at $380K.",
            "Customer-facing comms on hold pending IR.",
            "24-hour containment window per policy.",
            "MITRE mapping updated in the ticket.",
            "Insurance carrier notified at T+6h.",
            "External IR firm retained as of yesterday.",
            "Board cyber committee briefed in writing.",
        ],
    }.get(domain, [""])
    return pool[h % len(pool)]


def _stylize(text: str, turn: int, domain: str) -> str:
    """Wrap the structural claim in a realistic prose frame."""
    frame = _frame(turn, domain)
    salt = _salt(turn, domain)
    # Lower-case first char of body for natural flow after the frame.
    body = text
    if body and body[0].isupper() and not body.startswith((
            "A ", "An ", "The ", "In ", "On ", "With ", "For ", "By ",
            "At ", "Per ", "As ", "From ", "It ", "EU-", "OR ", "ICU",
            "MRI", "CT ", "ID ", "US ", "EU ")):
        body = body[0].lower() + body[1:]
    return f"{frame} {body} {salt}".strip()


def _node(nid: str, ntype: str, role: str) -> Dict[str, Any]:
    return {"id": nid, "type": ntype, "role": role}


def _edge(src: str, dst: str, et: str = "causes",
          scope: str | None = None) -> Dict[str, Any]:
    e = {"src": src, "dst": dst, "type": et}
    if scope:
        e["scope"] = scope
    return e


# ---- Domain A: supply chain (20 interactions) -------------------------


def _supply_chain(e: Dict[str, str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    # Block 1 (1-4): Asian port congestion cascade
    items.append({
        "domain": "supply_chain", "turn_local": 1,
        "text": _sub(e, "Congestion at {{port_asia}} since the weekend — "
                        "ships are anchoring 5-7 days out; container "
                        "throughput is down roughly 40% week-over-week."),
        "fragment": {
            "nodes": [_node("port_congestion", "state", "root"),
                      _node("container_throughput_drop", "consequence", "terminal")],
            "edges": [_edge("port_congestion", "container_throughput_drop")],
        },
    })
    items.append({
        "domain": "supply_chain", "turn_local": 2,
        "text": _sub(e, "The {{port_asia}} slowdown is now cascading — "
                        "arrivals into {{port_us}} are 8 days behind plan, "
                        "and the {{warehouse_us}} inbound lane has 11 empty "
                        "racks as of this morning."),
        "fragment": {
            "nodes": [_node(e["port_asia"], "entity", "root"),
                      _node("port_backup", "consequence", "intermediate"),
                      _node(e["warehouse_us"], "entity", "intermediate"),
                      _node("empty_racks", "state", "terminal")],
            "edges": [_edge(e["port_asia"], "port_backup"),
                      _edge("port_backup", e["warehouse_us"]),
                      _edge(e["warehouse_us"], "empty_racks")],
        },
    })
    items.append({
        "domain": "supply_chain", "turn_local": 3,
        "text": _sub(e, "Low stock at {{warehouse_us}} meant {{retailer}} "
                        "couldn't restock best-sellers, leading to stockouts "
                        "at store level."),
        "fragment": {
            "nodes": [_node(e["warehouse_us"], "entity", "root"),
                      _node("low_stock", "state", "intermediate"),
                      _node(e["retailer"], "entity", "intermediate"),
                      _node("store_stockout", "consequence", "terminal")],
            "edges": [_edge(e["warehouse_us"], "low_stock"),
                      _edge("low_stock", e["retailer"]),
                      _edge(e["retailer"], "store_stockout")],
        },
    })
    items.append({
        "domain": "supply_chain", "turn_local": 4,
        "text": _sub(e, "The stockouts caused a sharp revenue miss at "
                        "{{retailer}}, prompting an emergency air-freight "
                        "program."),
        "fragment": {
            "nodes": [_node("store_stockout", "consequence", "root"),
                      _node("revenue_miss", "consequence", "intermediate"),
                      _node("emergency_air_freight", "event", "terminal")],
            "edges": [_edge("store_stockout", "revenue_miss"),
                      _edge("revenue_miss", "emergency_air_freight")],
        },
    })

    # Block 2 (5-8): Chip shortage cascade
    items.append({
        "domain": "supply_chain", "turn_local": 5,
        "text": _sub(e, "A {{disruptor_asia}} forced {{fab_origin}} to pause "
                        "wafer fabrication for ten days."),
        "fragment": {
            "nodes": [_node(e["disruptor_asia"], "event", "root"),
                      _node("fab_pause", "consequence", "terminal")],
            "edges": [_edge(e["disruptor_asia"], "fab_pause")],
        },
    })
    items.append({
        "domain": "supply_chain", "turn_local": 6,
        "text": _sub(e, "The fab pause triggered a global semiconductor "
                        "shortage and pushed chip spot prices to record highs."),
        "fragment": {
            "nodes": [_node("fab_pause", "state", "root"),
                      _node("chip_shortage", "state", "intermediate"),
                      _node("chip_spot_price_up", "consequence", "terminal")],
            "edges": [_edge("fab_pause", "chip_shortage"),
                      _edge("chip_shortage", "chip_spot_price_up")],
        },
    })
    items.append({
        "domain": "supply_chain", "turn_local": 7,
        "text": _sub(e, "Chip shortages delayed consumer electronics assembly "
                        "and forced device retail prices higher."),
        "fragment": {
            "nodes": [_node("chip_shortage", "state", "root"),
                      _node("electronics_assembly_delay", "consequence", "intermediate"),
                      _node("device_retail_up", "consequence", "terminal")],
            "edges": [_edge("chip_shortage", "electronics_assembly_delay"),
                      _edge("electronics_assembly_delay", "device_retail_up")],
        },
    })
    items.append({
        "domain": "supply_chain", "turn_local": 8,
        "text": _sub(e, "Higher device prices cut promo discounts and reduced "
                        "holiday season unit volumes."),
        "fragment": {
            "nodes": [_node("device_retail_up", "state", "root"),
                      _node("promo_cut", "consequence", "intermediate"),
                      _node("holiday_volume_down", "consequence", "terminal")],
            "edges": [_edge("device_retail_up", "promo_cut"),
                      _edge("promo_cut", "holiday_volume_down")],
        },
    })

    # Block 3 (9-12): European trucker strike cascade + {{city_eu}} as hub
    items.append({
        "domain": "supply_chain", "turn_local": 9,
        "text": _sub(e, "Drivers on the {{route_eu}} ground-freight corridor "
                        "went on strike, halting automotive parts shipments."),
        "fragment": {
            "nodes": [_node("driver_strike", "event", "root"),
                      _node("freight_halt", "consequence", "intermediate"),
                      _node("auto_parts_delay", "consequence", "terminal")],
            "edges": [_edge("driver_strike", "freight_halt"),
                      _edge("freight_halt", "auto_parts_delay")],
        },
    })
    items.append({
        "domain": "supply_chain", "turn_local": 10,
        "text": _sub(e, "Auto-parts shortage pushed {{city_eu}} assembly "
                        "plants to slow production and miss dealer targets."),
        "fragment": {
            "nodes": [_node("auto_parts_delay", "consequence", "root"),
                      _node(e["city_eu"] + "_assembly", "entity", "intermediate"),
                      _node("dealer_target_miss", "consequence", "terminal")],
            "edges": [_edge("auto_parts_delay", e["city_eu"] + "_assembly"),
                      _edge(e["city_eu"] + "_assembly", "dealer_target_miss")],
        },
    })
    items.append({
        "domain": "supply_chain", "turn_local": 11,
        "text": _sub(e, "Dealer shortages at {{city_eu}} triggered emergency "
                        "air freight from Asian suppliers, doubling logistics "
                        "cost."),
        "fragment": {
            "nodes": [_node("dealer_target_miss", "consequence", "root"),
                      _node("air_freight_surge", "event", "intermediate"),
                      _node("logistics_cost_up", "consequence", "terminal")],
            "edges": [_edge("dealer_target_miss", "air_freight_surge"),
                      _edge("air_freight_surge", "logistics_cost_up")],
        },
    })
    items.append({
        "domain": "supply_chain", "turn_local": 12,
        "text": _sub(e, "Customs clearance in {{city_eu}} requires an "
                        "EU-origin certificate for vehicles; domestic shipments "
                        "need only a commercial invoice."),
        "fragment": {
            "nodes": [_node(e["city_eu"], "role", "scope"),
                      _node("customs_clearance", "requirement", "intermediate"),
                      _node("eu_origin_cert", "credential", "terminal"),
                      _node("commercial_invoice", "credential", "intermediate")],
            "edges": [_edge("customs_clearance", "eu_origin_cert",
                            "requires", scope=e["city_eu"]),
                      _edge("customs_clearance", "commercial_invoice", "requires")],
        },
    })

    # Block 4 (13-16): Agricultural cascade
    items.append({
        "domain": "supply_chain", "turn_local": 13,
        "text": _sub(e, "A {{disruptor_agri}} sharply reduced {{commodity_agri}} "
                        "yields in the main growing region."),
        "fragment": {
            "nodes": [_node(e["disruptor_agri"], "event", "root"),
                      _node(e["commodity_agri"] + "_yield_drop",
                            "consequence", "terminal")],
            "edges": [_edge(e["disruptor_agri"], e["commodity_agri"] + "_yield_drop")],
        },
    })
    items.append({
        "domain": "supply_chain", "turn_local": 14,
        "text": _sub(e, "Lower {{commodity_agri}} supply drove wholesale "
                        "prices up, which lifted costs at bakeries and food "
                        "manufacturers."),
        "fragment": {
            "nodes": [_node(e["commodity_agri"] + "_yield_drop", "state", "root"),
                      _node("wholesale_up", "consequence", "intermediate"),
                      _node("bakery_cost_up", "consequence", "terminal")],
            "edges": [_edge(e["commodity_agri"] + "_yield_drop", "wholesale_up"),
                      _edge("wholesale_up", "bakery_cost_up")],
        },
    })
    items.append({
        "domain": "supply_chain", "turn_local": 15,
        "text": _sub(e, "Bakeries passed the higher costs to consumers as "
                        "{{consumer_product_a}} and {{consumer_product_b}} "
                        "prices rose in grocery stores."),
        "fragment": {
            "nodes": [_node("bakery_cost_up", "state", "root"),
                      _node(e["consumer_product_a"] + "_price_up",
                            "consequence", "terminal"),
                      _node(e["consumer_product_b"] + "_price_up",
                            "consequence", "terminal")],
            "edges": [_edge("bakery_cost_up", e["consumer_product_a"] + "_price_up"),
                      _edge("bakery_cost_up", e["consumer_product_b"] + "_price_up")],
        },
    })
    items.append({
        "domain": "supply_chain", "turn_local": 16,
        "text": _sub(e, "Rising {{consumer_product_a}} prices reduced "
                        "household discretionary spend on other categories."),
        "fragment": {
            "nodes": [_node(e["consumer_product_a"] + "_price_up",
                            "state", "root"),
                      _node("discretionary_down", "consequence", "terminal")],
            "edges": [_edge(e["consumer_product_a"] + "_price_up",
                            "discretionary_down")],
        },
    })

    # Block 5 (17-20): European export bottleneck — {{port_eu}} as hub
    items.append({
        "domain": "supply_chain", "turn_local": 17,
        "text": _sub(e, "Container shortages at {{port_eu}} slowed European "
                        "exports bound for North America."),
        "fragment": {
            "nodes": [_node(e["port_eu"], "entity", "root"),
                      _node("container_shortage", "state", "intermediate"),
                      _node("eu_export_slowdown", "consequence", "terminal")],
            "edges": [_edge(e["port_eu"], "container_shortage"),
                      _edge("container_shortage", "eu_export_slowdown")],
        },
    })
    items.append({
        "domain": "supply_chain", "turn_local": 18,
        "text": _sub(e, "Slower EU exports reduced specialty food availability "
                        "at US retailers, forcing domestic substitution."),
        "fragment": {
            "nodes": [_node("eu_export_slowdown", "consequence", "root"),
                      _node("specialty_food_short", "state", "intermediate"),
                      _node("domestic_substitution", "event", "terminal")],
            "edges": [_edge("eu_export_slowdown", "specialty_food_short"),
                      _edge("specialty_food_short", "domestic_substitution")],
        },
    })
    items.append({
        "domain": "supply_chain", "turn_local": 19,
        "text": _sub(e, "Specialty retailers renegotiated supplier contracts "
                        "and added a logistics risk premium to next year's "
                        "pricing."),
        "fragment": {
            "nodes": [_node("domestic_substitution", "event", "root"),
                      _node("contract_renegotiation", "event", "intermediate"),
                      _node("risk_premium", "consequence", "terminal")],
            "edges": [_edge("domestic_substitution", "contract_renegotiation"),
                      _edge("contract_renegotiation", "risk_premium")],
        },
    })
    items.append({
        "domain": "supply_chain", "turn_local": 20,
        "text": _sub(e, "Rising logistics cost across these cascades lifted "
                        "CPG retail prices and further reduced discretionary "
                        "spending."),
        "fragment": {
            "nodes": [_node("logistics_cost_up", "state", "root"),
                      _node("cpg_price_up", "consequence", "intermediate"),
                      _node("discretionary_down", "consequence", "terminal")],
            "edges": [_edge("logistics_cost_up", "cpg_price_up"),
                      _edge("cpg_price_up", "discretionary_down")],
        },
    })
    return items


# ---- Domain B: hospital operations (20 interactions) ------------------


def _hospital(e: Dict[str, str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    # Block 1 (1-5): ER overflow cascade
    items.append({
        "domain": "hospital", "turn_local": 1,
        "text": _sub(e, "A severe flu wave overwhelmed the {{er}} at "
                        "{{hospital}}, pushing wait times past six hours."),
        "fragment": {
            "nodes": [_node("flu_wave", "event", "root"),
                      _node(e["er"] + "_overflow", "state", "intermediate"),
                      _node("six_hour_wait", "consequence", "terminal")],
            "edges": [_edge("flu_wave", e["er"] + "_overflow"),
                      _edge(e["er"] + "_overflow", "six_hour_wait")],
        },
    })
    items.append({
        "domain": "hospital", "turn_local": 2,
        "text": _sub(e, "Triage was delayed and non-critical patients were "
                        "diverted to nearby urgent care."),
        "fragment": {
            "nodes": [_node("six_hour_wait", "state", "root"),
                      _node("triage_delay", "consequence", "intermediate"),
                      _node("urgent_care_diversion", "event", "terminal")],
            "edges": [_edge("six_hour_wait", "triage_delay"),
                      _edge("triage_delay", "urgent_care_diversion")],
        },
    })
    items.append({
        "domain": "hospital", "turn_local": 3,
        "text": _sub(e, "The {{icu}} filled as respiratory admissions rose, "
                        "and elective admissions were put on hold."),
        "fragment": {
            "nodes": [_node("respiratory_admits", "event", "root"),
                      _node(e["icu"] + "_full", "state", "intermediate"),
                      _node("elective_hold", "event", "terminal")],
            "edges": [_edge("respiratory_admits", e["icu"] + "_full"),
                      _edge(e["icu"] + "_full", "elective_hold")],
        },
    })
    items.append({
        "domain": "hospital", "turn_local": 4,
        "text": _sub(e, "Held electives triggered surgical scheduling "
                        "cascade — rescheduled cases rolled into the next "
                        "week, straining operating-room capacity."),
        "fragment": {
            "nodes": [_node("elective_hold", "state", "root"),
                      _node("surgical_reschedule", "event", "intermediate"),
                      _node("or_capacity_strain", "consequence", "terminal")],
            "edges": [_edge("elective_hold", "surgical_reschedule"),
                      _edge("surgical_reschedule", "or_capacity_strain")],
        },
    })
    items.append({
        "domain": "hospital", "turn_local": 5,
        "text": _sub(e, "OR capacity strain led to staff burnout and a "
                        "temporary locum hiring push."),
        "fragment": {
            "nodes": [_node("or_capacity_strain", "state", "root"),
                      _node("staff_burnout", "consequence", "intermediate"),
                      _node("locum_hiring", "event", "terminal")],
            "edges": [_edge("or_capacity_strain", "staff_burnout"),
                      _edge("staff_burnout", "locum_hiring")],
        },
    })

    # Block 2 (6-10): Equipment bottleneck cascade
    items.append({
        "domain": "hospital", "turn_local": 6,
        "text": _sub(e, "The main {{equipment_key}} in {{dept_rad}} went "
                        "down, creating a radiology backlog."),
        "fragment": {
            "nodes": [_node(e["equipment_key"] + "_down", "event", "root"),
                      _node(e["dept_rad"] + "_backlog", "state", "terminal")],
            "edges": [_edge(e["equipment_key"] + "_down",
                            e["dept_rad"] + "_backlog")],
        },
    })
    items.append({
        "domain": "hospital", "turn_local": 7,
        "text": _sub(e, "Radiology delays pushed surgeries back because "
                        "pre-op imaging wasn't available in time."),
        "fragment": {
            "nodes": [_node(e["dept_rad"] + "_backlog", "state", "root"),
                      _node("preop_image_delay", "consequence", "intermediate"),
                      _node(e["dept_surg"] + "_delay", "consequence", "terminal")],
            "edges": [_edge(e["dept_rad"] + "_backlog", "preop_image_delay"),
                      _edge("preop_image_delay", e["dept_surg"] + "_delay")],
        },
    })
    items.append({
        "domain": "hospital", "turn_local": 8,
        "text": _sub(e, "Surgery delays meant ICU beds stayed reserved for "
                        "post-op cases that never arrived; intake coordination "
                        "broke down."),
        "fragment": {
            "nodes": [_node(e["dept_surg"] + "_delay", "state", "root"),
                      _node("icu_bed_reserved", "state", "intermediate"),
                      _node("intake_breakdown", "consequence", "terminal")],
            "edges": [_edge(e["dept_surg"] + "_delay", "icu_bed_reserved"),
                      _edge("icu_bed_reserved", "intake_breakdown")],
        },
    })
    items.append({
        "domain": "hospital", "turn_local": 9,
        "text": _sub(e, "Intake breakdown produced billing inaccuracies and "
                        "patient satisfaction scores dropped."),
        "fragment": {
            "nodes": [_node("intake_breakdown", "state", "root"),
                      _node("billing_errors", "consequence", "intermediate"),
                      _node("patient_sat_drop", "consequence", "terminal")],
            "edges": [_edge("intake_breakdown", "billing_errors"),
                      _edge("billing_errors", "patient_sat_drop")],
        },
    })
    items.append({
        "domain": "hospital", "turn_local": 10,
        "text": _sub(e, "The incident review recommended backup {{equipment_key}} "
                        "capacity, a cost the CFO flagged as too high."),
        "fragment": {
            "nodes": [_node("patient_sat_drop", "state", "root"),
                      _node("incident_review", "event", "intermediate"),
                      _node("backup_equipment_req", "requirement", "terminal"),
                      _node("cfo_pushback", "consequence", "terminal")],
            "edges": [_edge("patient_sat_drop", "incident_review"),
                      _edge("incident_review", "backup_equipment_req"),
                      _edge("backup_equipment_req", "cfo_pushback", "blocks")],
        },
    })

    # Block 3 (11-15): Protocol scope dependencies
    items.append({
        "domain": "hospital", "turn_local": 11,
        "text": _sub(e, "A {{role_junior}} cannot complete a "
                        "{{action_controlled}} without cosign from a "
                        "{{role_senior}}, but can authorize "
                        "{{action_routine}} alone."),
        "fragment": {
            "nodes": [_node(e["role_junior"], "role", "scope"),
                      _node(e["action_controlled"], "requirement", "intermediate"),
                      _node(e["role_senior"] + "_cosign", "prerequisite", "terminal"),
                      _node(e["action_routine"], "requirement", "intermediate")],
            "edges": [_edge(e["action_controlled"],
                            e["role_senior"] + "_cosign",
                            "requires", scope=e["role_junior"]),
                      _edge(e["action_routine"], e["role_junior"], "enables",
                            scope=e["role_junior"])],
        },
    })
    items.append({
        "domain": "hospital", "turn_local": 12,
        "text": _sub(e, "Pediatric protocols require dosing by weight; "
                        "adult protocols use fixed dosing."),
        "fragment": {
            "nodes": [_node("pediatric", "role", "scope"),
                      _node("adult", "role", "scope"),
                      _node("dosing", "requirement", "intermediate"),
                      _node("weight_based_dose", "prerequisite", "terminal"),
                      _node("fixed_dose", "prerequisite", "terminal")],
            "edges": [_edge("dosing", "weight_based_dose",
                            "requires", scope="pediatric"),
                      _edge("dosing", "fixed_dose",
                            "requires", scope="adult")],
        },
    })
    items.append({
        "domain": "hospital", "turn_local": 13,
        "text": _sub(e, "In {{icu}}, central line placement requires sterile "
                        "procedure sign-off; in general wards, a standard "
                        "time-out is enough."),
        "fragment": {
            "nodes": [_node(e["icu"], "role", "scope"),
                      _node("general_ward", "role", "scope"),
                      _node("central_line", "requirement", "intermediate"),
                      _node("sterile_signoff", "prerequisite", "terminal"),
                      _node("time_out_check", "prerequisite", "terminal")],
            "edges": [_edge("central_line", "sterile_signoff",
                            "requires", scope=e["icu"]),
                      _edge("central_line", "time_out_check",
                            "requires", scope="general_ward")],
        },
    })
    items.append({
        "domain": "hospital", "turn_local": 14,
        "text": _sub(e, "Discharge from {{er}} requires attending review; "
                        "discharge from primary care is done by the "
                        "supervising physician."),
        "fragment": {
            "nodes": [_node(e["er"], "role", "scope"),
                      _node("primary_care", "role", "scope"),
                      _node("discharge", "requirement", "intermediate"),
                      _node(e["role_senior"] + "_review", "prerequisite", "terminal"),
                      _node("supervising_physician", "role", "terminal")],
            "edges": [_edge("discharge", e["role_senior"] + "_review",
                            "requires", scope=e["er"]),
                      _edge("discharge", "supervising_physician",
                            "requires", scope="primary_care")],
        },
    })
    items.append({
        "domain": "hospital", "turn_local": 15,
        "text": _sub(e, "Nursing protocols forbid a {{role_junior}} from "
                        "administering chemotherapy independently; all "
                        "chemotherapy requires attending oversight."),
        "fragment": {
            "nodes": [_node(e["role_junior"], "role", "scope"),
                      _node("chemo_admin", "requirement", "intermediate"),
                      _node("attending_oversight", "prerequisite", "terminal")],
            "edges": [_edge("chemo_admin", "attending_oversight",
                            "requires", scope=e["role_junior"])],
        },
    })

    # Block 4 (16-20): Supply cascade
    items.append({
        "domain": "hospital", "turn_local": 16,
        "text": _sub(e, "A {{supply_consumable}} shortage hit after the "
                        "primary supplier recalled a lot, and surgical kits "
                        "ran short."),
        "fragment": {
            "nodes": [_node(e["supply_consumable"] + "_recall", "event", "root"),
                      _node("surgical_kit_short", "state", "terminal")],
            "edges": [_edge(e["supply_consumable"] + "_recall",
                            "surgical_kit_short")],
        },
    })
    items.append({
        "domain": "hospital", "turn_local": 17,
        "text": _sub(e, "Kit shortage caused elective procedures to be "
                        "rescheduled and emergency bulk purchasing was "
                        "approved."),
        "fragment": {
            "nodes": [_node("surgical_kit_short", "state", "root"),
                      _node("elective_reschedule", "event", "intermediate"),
                      _node("bulk_purchase", "event", "terminal")],
            "edges": [_edge("surgical_kit_short", "elective_reschedule"),
                      _edge("elective_reschedule", "bulk_purchase")],
        },
    })
    items.append({
        "domain": "hospital", "turn_local": 18,
        "text": _sub(e, "Bulk purchases went through at spot prices, "
                        "increasing hospital supply costs for the quarter."),
        "fragment": {
            "nodes": [_node("bulk_purchase", "event", "root"),
                      _node("supply_cost_up", "consequence", "terminal")],
            "edges": [_edge("bulk_purchase", "supply_cost_up")],
        },
    })
    items.append({
        "domain": "hospital", "turn_local": 19,
        "text": _sub(e, "Supply cost increases forced the finance committee "
                        "to defer a planned wage adjustment for support "
                        "staff."),
        "fragment": {
            "nodes": [_node("supply_cost_up", "state", "root"),
                      _node("wage_deferral", "event", "terminal")],
            "edges": [_edge("supply_cost_up", "wage_deferral")],
        },
    })
    items.append({
        "domain": "hospital", "turn_local": 20,
        "text": _sub(e, "Wage deferral accelerated resignations among "
                        "experienced staff, worsening the {{er}} overflow "
                        "we saw earlier."),
        "fragment": {
            "nodes": [_node("wage_deferral", "state", "root"),
                      _node("staff_resignation", "consequence", "intermediate"),
                      _node(e["er"] + "_overflow", "state", "terminal")],
            "edges": [_edge("wage_deferral", "staff_resignation"),
                      _edge("staff_resignation", e["er"] + "_overflow")],
        },
    })
    return items


# ---- Domain C: cybersecurity (10 interactions) ------------------------


def _cybersecurity(e: Dict[str, str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    items.append({
        "domain": "cybersecurity", "turn_local": 1,
        "text": _sub(e, "An unpatched exposure in the {{entry_point}} was "
                        "exploited by {{attacker}}, giving initial access."),
        "fragment": {
            "nodes": [_node(e["entry_point"], "vulnerability", "root"),
                      _node(e["attacker"], "event", "intermediate"),
                      _node("initial_access", "state", "terminal")],
            "edges": [_edge(e["entry_point"], e["attacker"]),
                      _edge(e["attacker"], "initial_access")],
        },
    })
    items.append({
        "domain": "cybersecurity", "turn_local": 2,
        "text": _sub(e, "The attacker moved laterally from initial access to "
                        "the {{lateral_target}} and stole privileged "
                        "credentials."),
        "fragment": {
            "nodes": [_node("initial_access", "state", "root"),
                      _node("lateral_movement", "event", "intermediate"),
                      _node(e["lateral_target"], "entity", "intermediate"),
                      _node("cred_theft", "event", "terminal")],
            "edges": [_edge("initial_access", "lateral_movement"),
                      _edge("lateral_movement", e["lateral_target"]),
                      _edge(e["lateral_target"], "cred_theft")],
        },
    })
    items.append({
        "domain": "cybersecurity", "turn_local": 3,
        "text": _sub(e, "With privileged credentials, the {{terminal_system}} "
                        "was encrypted; this knocked out {{critical_service}}."),
        "fragment": {
            "nodes": [_node("cred_theft", "event", "root"),
                      _node(e["terminal_system"] + "_encrypted",
                            "consequence", "intermediate"),
                      _node(e["critical_service"] + "_down",
                            "consequence", "terminal")],
            "edges": [_edge("cred_theft",
                            e["terminal_system"] + "_encrypted"),
                      _edge(e["terminal_system"] + "_encrypted",
                            e["critical_service"] + "_down")],
        },
    })
    items.append({
        "domain": "cybersecurity", "turn_local": 4,
        "text": _sub(e, "With {{critical_service}} down, dependent workflows "
                        "queued and customer complaints spiked."),
        "fragment": {
            "nodes": [_node(e["critical_service"] + "_down", "state", "root"),
                      _node("workflow_queue", "consequence", "intermediate"),
                      _node("customer_complaint", "consequence", "terminal")],
            "edges": [_edge(e["critical_service"] + "_down", "workflow_queue"),
                      _edge("workflow_queue", "customer_complaint")],
        },
    })
    items.append({
        "domain": "cybersecurity", "turn_local": 5,
        "text": _sub(e, "Incident response pulled the affected segment "
                        "offline, and a ransom demand arrived; the board was "
                        "briefed."),
        "fragment": {
            "nodes": [_node("customer_complaint", "state", "root"),
                      _node("segment_offline", "event", "intermediate"),
                      _node("ransom_demand", "event", "intermediate"),
                      _node("board_brief", "event", "terminal")],
            "edges": [_edge("customer_complaint", "segment_offline"),
                      _edge("segment_offline", "ransom_demand"),
                      _edge("ransom_demand", "board_brief")],
        },
    })

    # Block 2 (6-10): Access tier scope dependencies
    items.append({
        "domain": "cybersecurity", "turn_local": 6,
        "text": _sub(e, "A {{role_analyst_junior}} can acknowledge low-severity "
                        "alerts alone but cannot authorize a "
                        "{{action_cyber_controlled}} without a "
                        "{{role_analyst_senior}} approval."),
        "fragment": {
            "nodes": [_node(e["role_analyst_junior"], "role", "scope"),
                      _node("low_sev_ack", "requirement", "intermediate"),
                      _node(e["action_cyber_controlled"], "requirement",
                            "intermediate"),
                      _node(e["role_analyst_senior"] + "_approval",
                            "prerequisite", "terminal")],
            "edges": [_edge(e["action_cyber_controlled"],
                            e["role_analyst_senior"] + "_approval",
                            "requires", scope=e["role_analyst_junior"])],
        },
    })
    items.append({
        "domain": "cybersecurity", "turn_local": 7,
        "text": _sub(e, "Emergency change windows bypass the senior-approval "
                        "requirement, but require a CISO sign-off after the "
                        "fact."),
        "fragment": {
            "nodes": [_node("emergency_change", "role", "scope"),
                      _node(e["action_cyber_controlled"], "requirement",
                            "intermediate"),
                      _node("ciso_post_signoff", "prerequisite", "terminal")],
            "edges": [_edge(e["action_cyber_controlled"], "ciso_post_signoff",
                            "requires", scope="emergency_change")],
        },
    })
    items.append({
        "domain": "cybersecurity", "turn_local": 8,
        "text": _sub(e, "A routine misconfiguration on the {{terminal_system}} "
                        "triggered duplicate alerts, and analyst fatigue "
                        "slowed triage."),
        "fragment": {
            "nodes": [_node("misconfig", "event", "root"),
                      _node("duplicate_alerts", "consequence", "intermediate"),
                      _node("analyst_fatigue", "consequence", "intermediate"),
                      _node("triage_slow", "consequence", "terminal")],
            "edges": [_edge("misconfig", "duplicate_alerts"),
                      _edge("duplicate_alerts", "analyst_fatigue"),
                      _edge("analyst_fatigue", "triage_slow")],
        },
    })
    items.append({
        "domain": "cybersecurity", "turn_local": 9,
        "text": _sub(e, "Slow triage meant a real intrusion through "
                        "{{entry_point}} sat undetected for 36 hours before "
                        "escalation."),
        "fragment": {
            "nodes": [_node("triage_slow", "state", "root"),
                      _node(e["entry_point"] + "_unnoticed", "state",
                            "intermediate"),
                      _node("delayed_escalation", "consequence", "terminal")],
            "edges": [_edge("triage_slow", e["entry_point"] + "_unnoticed"),
                      _edge(e["entry_point"] + "_unnoticed",
                            "delayed_escalation")],
        },
    })
    items.append({
        "domain": "cybersecurity", "turn_local": 10,
        "text": _sub(e, "Delayed escalation increased damage scope; the "
                        "business impact forced a full incident post-mortem "
                        "and new tooling spend."),
        "fragment": {
            "nodes": [_node("delayed_escalation", "state", "root"),
                      _node("damage_expansion", "consequence", "intermediate"),
                      _node("postmortem", "event", "intermediate"),
                      _node("tooling_spend", "event", "terminal")],
            "edges": [_edge("delayed_escalation", "damage_expansion"),
                      _edge("damage_expansion", "postmortem"),
                      _edge("postmortem", "tooling_spend")],
        },
    })
    return items


# ----------------------------------------------------------------------
# Cross-domain queries (10)
# ----------------------------------------------------------------------


def _queries(e: Dict[str, str]) -> List[Dict[str, Any]]:
    q: List[Dict[str, Any]] = []

    # Type 1 (recall, 3 queries) — answers require nodes from turn 2-18
    q.append({
        "qid": "q1_recall_shanghai",
        "type": "recall",
        "text": _sub(e, "Months back we traced problems at {{warehouse_us}} "
                        "upstream. Given everything we've discussed, what was "
                        "the furthest upstream cause that affected "
                        "{{warehouse_us}}?"),
        "expected_answer_nodes": [e["port_asia"], "port_backup"],
        "expected_answer_text": _sub(e,
            "The furthest upstream cause was port congestion at "
            "{{port_asia}} that backed up arrivals into {{warehouse_us}}."),
        "requires": "h_state recall of turn 1-2 supply-chain nodes",
        "anchor_nodes": [e["warehouse_us"]],
    })
    q.append({
        "qid": "q2_recall_chipshortage",
        "type": "recall",
        "text": _sub(e, "A distributor is asking us why holiday electronics "
                        "volumes were lower this year. What chain did we "
                        "trace earlier that led to lower holiday volume?"),
        "expected_answer_nodes": [e["disruptor_asia"], "fab_pause",
                                   "chip_shortage", "device_retail_up",
                                   "promo_cut", "holiday_volume_down"],
        "expected_answer_text": _sub(e,
            "The chain was: {{disruptor_asia}} caused a fab pause, "
            "which produced a chip shortage, raised device retail prices, "
            "cut promotional discounts, and reduced holiday unit volume."),
        "requires": "h_state recall of supply-chain chip cascade",
        "anchor_nodes": ["holiday_volume_down"],
    })
    q.append({
        "qid": "q3_recall_equipment",
        "type": "recall",
        "text": _sub(e, "The CFO rejected a backup-equipment capital request "
                        "earlier. What chain of events led to that request?"),
        "expected_answer_nodes": [e["equipment_key"] + "_down",
                                   "preop_image_delay",
                                   "intake_breakdown",
                                   "incident_review",
                                   "backup_equipment_req"],
        "expected_answer_text": _sub(e,
            "{{equipment_key}} went down, backlog formed in {{dept_rad}}, "
            "pre-op imaging delayed {{dept_surg}}, intake broke down, "
            "patient satisfaction dropped, and the incident review produced "
            "the backup-equipment requirement."),
        "requires": "h_state recall of hospital equipment cascade",
        "anchor_nodes": ["cfo_pushback", "backup_equipment_req"],
    })

    # Type 2 (analogy, 3 queries) — require pattern library match across domains
    q.append({
        "qid": "q4_analogy_ransomware",
        "type": "analogy",
        "text": _sub(e, "We just described a ransomware incident that went "
                        "{{entry_point}} → {{lateral_target}} → encrypted "
                        "systems → {{critical_service}} offline. Have we "
                        "discussed any structurally similar chain in another "
                        "domain?"),
        "expected_answer_nodes": [e["port_asia"], e["disruptor_asia"],
                                   e["equipment_key"] + "_down"],
        "expected_answer_text":
            "Yes — this is the same cascade-failure shape as the supply-chain "
            "cascades (port congestion → logistics impact → downstream retail) "
            "and the hospital equipment cascade "
            "(equipment down → department backlog → intake breakdown).",
        "requires": "pattern library cross-domain match",
        "anchor_nodes": [e["attacker"], "lateral_movement",
                         e["terminal_system"] + "_encrypted"],
    })
    q.append({
        "qid": "q5_analogy_scope",
        "type": "analogy",
        "text": _sub(e, "In the hospital, {{role_junior}} can't do "
                        "{{action_controlled}} without {{role_senior}} cosign. "
                        "Across what we've discussed, what other scope-bounded "
                        "authorization patterns are similar?"),
        "expected_answer_nodes": [e["role_analyst_junior"],
                                   e["action_cyber_controlled"],
                                   e["role_analyst_senior"] + "_approval",
                                   "customs_clearance", "eu_origin_cert"],
        "expected_answer_text":
            "Similar scope-gated authorization in cybersecurity (tier-1 "
            "analyst can't do a production change without senior approval) "
            "and in supply-chain customs (EU-origin certificate required "
            "only for shipments crossing the EU border).",
        "requires": "pattern library scope match",
        "anchor_nodes": [e["action_controlled"],
                         e["role_senior"] + "_cosign"],
    })
    q.append({
        "qid": "q6_analogy_bottleneck",
        "type": "analogy",
        "text": _sub(e, "The {{equipment_key}} is a single point of failure "
                        "for imaging. What other single-point-of-failure "
                        "entities have we encountered across the supply-chain "
                        "and cybersecurity domains?"),
        "expected_answer_nodes": [e["port_asia"], e["port_eu"],
                                   e["lateral_target"], "cred_theft"],
        "expected_answer_text":
            "Port congestion at the Asian and European port hubs acts as a "
            "single point of failure for supply-chain flow, and the domain "
            "controller/credential theft path is the equivalent single point "
            "of failure in cybersecurity.",
        "requires": "metric-centrality analysis across domains",
        "anchor_nodes": [e["equipment_key"] + "_down"],
    })

    # Type 3 (topology, 2 queries) — require global graph analysis
    q.append({
        "qid": "q7_topology_hubs",
        "type": "topology",
        "text": _sub(e, "Across everything we've discussed — supply chain, "
                        "hospital, cybersecurity — what are the top single "
                        "points of failure? Which entities, if disrupted, "
                        "cascade into the most downstream consequences?"),
        "expected_answer_nodes": [e["port_asia"], e["equipment_key"] + "_down",
                                   e["lateral_target"], e["port_eu"]],
        "expected_answer_text":
            "The highest-centrality hub nodes are: the Asian port hub "
            "(supply-chain origin), the European port hub, the hospital's "
            "critical imaging equipment, and the cybersecurity lateral target.",
        "requires": "full h_state centrality across all domains",
        "anchor_nodes": [],
    })
    q.append({
        "qid": "q8_topology_clusters",
        "type": "topology",
        "text": _sub(e, "How many structurally distinct cascade 'shapes' have "
                        "we accumulated across all three domains, and what "
                        "do they share?"),
        "expected_answer_nodes": [],
        "expected_answer_text":
            "Three main cascade shapes: (1) transport/supply bottleneck → "
            "downstream shortage → consumer/customer impact; (2) equipment "
            "or resource failure → department backlog → service delay; "
            "(3) attack ingress → lateral movement → critical-service impact. "
            "All share a root-disruption → intermediate-consequence → "
            "terminal-impact topology.",
        "requires": "metric-cluster enumeration across full state",
        "anchor_nodes": [],
    })

    # Type 4 (scope transfer, 2 queries)
    q.append({
        "qid": "q9_scope_transfer_cyber",
        "type": "scope_transfer",
        "text": _sub(e, "In the hospital, a {{role_junior}} can't authorize "
                        "{{action_controlled}} without a cosign. We're "
                        "designing a cyber escalation tier. Should a "
                        "{{role_analyst_junior}} be able to authorize a "
                        "{{action_cyber_controlled}} without "
                        "{{role_analyst_senior}} approval?"),
        "expected_answer_nodes": [e["role_analyst_junior"],
                                   e["role_analyst_senior"] + "_approval"],
        "expected_answer_text":
            "No — the same scope-restricted pattern applies: a "
            "controlled/high-impact action should require senior approval. "
            "The structural rule from the hospital transfers directly.",
        "requires": "scope pattern transfer hospital → cyber",
        "anchor_nodes": [e["action_cyber_controlled"]],
    })
    q.append({
        "qid": "q10_scope_transfer_customs",
        "type": "scope_transfer",
        "text": _sub(e, "Supply-chain customs requires an EU-origin "
                        "certificate only for shipments into {{city_eu}}. "
                        "For a new domain that has {{disruptor_agri}} affecting "
                        "{{commodity_agri}} exports, do we need a scope-gated "
                        "certificate for US vs EU destinations?"),
        "expected_answer_nodes": ["customs_clearance", "eu_origin_cert"],
        "expected_answer_text":
            "Yes — for EU destinations apply the EU-origin certificate scope "
            "rule; for US destinations only the commercial invoice is needed. "
            "Same pattern we already defined for customs_clearance.",
        "requires": "scope pattern transfer within supply-chain",
        "anchor_nodes": [e["commodity_agri"] + "_yield_drop",
                         "customs_clearance"],
    })
    return q


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------


def _derive_topology_ground_truth(interactions: List[Dict[str, Any]], top_k: int = 6) -> Dict[str, Any]:
    """Compute the top-k downstream-reach SPOFs from the union of all
    fragments. Used to populate expected_answer_node_ids for topology
    queries so the structural scorer grades against what the ACTUAL
    graph says, not a priori human intuition."""
    import networkx as nx
    g = nx.DiGraph()
    node_types: Dict[str, str] = {}
    for it in interactions:
        for n in it["fragment"]["nodes"]:
            node_types[n["id"]] = n.get("type", "entity")
            g.add_node(n["id"])
        for ed in it["fragment"]["edges"]:
            if ed["src"] in g and ed["dst"] in g:
                g.add_edge(ed["src"], ed["dst"])
    reach = [(n, len(nx.descendants(g, n))) for n in g.nodes]
    reach.sort(key=lambda x: -x[1])
    top_spof = [n for n, r in reach[:top_k] if r > 0]
    # Cluster count = number of weakly-connected components that are chains
    n_components = nx.number_weakly_connected_components(g)
    return {
        "top_spof": top_spof,
        "n_components": n_components,
    }


def build_variant(variant_id: int) -> Dict[str, Any]:
    e = ENTITY_POOLS[variant_id]
    interactions = (_supply_chain(e) + _hospital(e) + _cybersecurity(e))
    # Tag absolute index + stylize every text into realistic operational prose
    for i, it in enumerate(interactions):
        it["turn"] = i + 1
        it["text_raw"] = it["text"]
        it["text"] = _stylize(it["text"], it["turn"], it["domain"])
    queries = _queries(e)
    for i, q in enumerate(queries):
        q["turn"] = 50 + i + 1

    # Populate structural ground truth: expected node IDs for each query.
    # For topology queries, compute the top-k downstream-reach nodes from
    # the actual graph. For all other types, use the already-authored
    # expected_answer_nodes list.
    topo = _derive_topology_ground_truth(interactions, top_k=6)
    for q in queries:
        if q["type"] == "topology":
            if "hubs" in q["qid"]:
                q["expected_answer_node_ids"] = topo["top_spof"]
                q["ground_truth_note"] = (
                    f"top-{len(topo['top_spof'])} SPOFs by downstream "
                    f"reach in the generated graph")
            elif "clusters" in q["qid"]:
                q["expected_answer_node_ids"] = []
                q["expected_cluster_count"] = topo["n_components"]
                q["ground_truth_note"] = (
                    f"{topo['n_components']} weakly-connected components "
                    "in the generated graph")
        else:
            # For recall/analogy/scope_transfer, promote the pre-authored
            # expected_answer_nodes into the grading key if present.
            q["expected_answer_node_ids"] = q.get("expected_answer_nodes", [])
    return {
        "variant_id": variant_id,
        "entities": e,
        "interactions": interactions,  # 50 items
        "queries": queries,             # 10 items
        "topology_ground_truth": topo,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True,
                   help="directory to write variant_{N}.json files")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for vid in range(len(ENTITY_POOLS)):
        variant = build_variant(vid)
        fn = out_dir / f"variant_{vid}.json"
        with open(fn, "w") as f:
            json.dump(variant, f, indent=2)
        manifest.append({
            "variant_id": vid,
            "path": str(fn),
            "n_interactions": len(variant["interactions"]),
            "n_queries": len(variant["queries"]),
        })
        print(f"wrote {fn} — {len(variant['interactions'])} interactions "
              f"+ {len(variant['queries'])} queries",
              flush=True)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
