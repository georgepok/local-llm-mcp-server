"""Hard test suite for Experiment 5 — designed to reliably break plain LLM.

Targets known failure modes of plain transformer reasoning:

  1. Deep chains (10-15 hops) — plain LLM stops at a plausible intermediate
  2. Interleaved multi-chain contamination — plain LLM conflates by vocabulary
  3. Multi-level scope conditioning — plain LLM misses the deepest scope
  4. Temporal reverse-order context — plain LLM anchors on text order
  5. Branching-merging DAGs — multiple paths to effect, root is not the salient one
  6. Lexical-decoy roots — a plausible non-root shares vocabulary with the question
  7. Long-distance scope-gated implication — scope conditions far from conclusion

Each test is deterministic and machine-gradeable via specific-keyword scoring.

Importable as TESTS_HARD; consumed by scripts/exp5_llm_integration.py via
--test_module argument (to be added).
"""

# Node type slots (align with liquid_arc TypeEmbed — any integer < n_node_types=32)
T_EVENT = 0
T_CONSEQUENCE = 1
T_STATE = 2
T_CAUSE = 3
T_ROLE = 4
T_CREDENTIAL = 5
T_REQUIREMENT = 6
T_PREREQUISITE = 7

R_ROOT = 0
R_INTER = 1
R_TERMINAL = 2
R_SCOPE = 3
R_QUERY = 4


def _chain(node_ids, node_types, edge_type=0):
    """Helper: build nodes + edges for a linear chain."""
    nodes = []
    for i, (nid, ntype) in enumerate(zip(node_ids, node_types)):
        if i == 0:
            role = R_ROOT
        elif i == len(node_ids) - 1:
            role = R_TERMINAL
        else:
            role = R_INTER
        nodes.append({'id': nid, 'type': ntype, 'role': role})
    edges = [{'src': a, 'dst': b, 'type': edge_type}
             for a, b in zip(node_ids, node_ids[1:])]
    return nodes, edges


TESTS_HARD = [
    # ════════════════════════════════════════════════════════════════
    # CATEGORY 1 — Deep chains (10+ hops)
    # ════════════════════════════════════════════════════════════════

    {
        'name': 'deep_chain_12hop_solar_storm',
        'type': 'root_cause',
        'context': (
            "A severe solar storm struck on Monday. The storm induced geomagnetic currents. "
            "The geomagnetic currents saturated high-voltage transformer cores. "
            "Saturated transformers tripped their protection relays. "
            "Tripped relays cascaded across the eastern grid. "
            "The grid collapse cut power to five states. "
            "Loss of power disabled data-center cooling. "
            "Overheated servers triggered emergency shutdowns. "
            "Shutdowns took the national healthcare records system offline. "
            "Offline records blocked electronic prescriptions. "
            "Blocked prescriptions stranded pharmacy queues. "
            "Stranded queues caused medicine rationing in hospitals."
        ),
        'question': "What was the ultimate root cause of the medicine rationing?",
        'ground_truth': 'solar storm',
        'plain_fail_keyword': 'blocked prescriptions',
        'graph': {
            'nodes': _chain(
                ['solar_storm','geomag_current','core_saturation','trip_relays',
                 'grid_collapse','power_loss','cooling_failure','server_shutdown',
                 'records_offline','prescription_block','pharmacy_queue','medicine_rationing'],
                [T_EVENT]*12
            )[0],
            'edges': _chain(
                ['solar_storm','geomag_current','core_saturation','trip_relays',
                 'grid_collapse','power_loss','cooling_failure','server_shutdown',
                 'records_offline','prescription_block','pharmacy_queue','medicine_rationing'],
                [T_EVENT]*12
            )[1],
        },
        'query': {'type': 'root_cause', 'target': 'medicine_rationing'},
    },

    {
        'name': 'deep_chain_11hop_regulation',
        'type': 'root_cause',
        'context': (
            "A new trade regulation was signed in January. The regulation raised tariffs on raw aluminum. "
            "Raised tariffs squeezed aluminum-dependent manufacturers. "
            "Squeezed manufacturers scaled back production. "
            "Production cuts reduced orders from bauxite miners. "
            "Reduced mining orders laid off miners in three provinces. "
            "Mass layoffs hit regional consumer spending. "
            "Lower spending closed local restaurants and services. "
            "Closed businesses emptied commercial real estate. "
            "Vacant commercial property triggered loan defaults. "
            "Loan defaults destabilized two regional banks."
        ),
        'question': "What ultimately caused the regional bank destabilization?",
        'ground_truth': 'trade regulation',
        'plain_fail_keyword': 'loan defaults',
        'graph': {
            'nodes': _chain(
                ['trade_regulation','tariff_raise','manufacturer_squeeze','production_cut',
                 'mining_cut','miner_layoffs','spending_drop','business_closure',
                 'vacancy','loan_default','bank_destabilization'],
                [T_EVENT]*11
            )[0],
            'edges': _chain(
                ['trade_regulation','tariff_raise','manufacturer_squeeze','production_cut',
                 'mining_cut','miner_layoffs','spending_drop','business_closure',
                 'vacancy','loan_default','bank_destabilization'],
                [T_EVENT]*11
            )[1],
        },
        'query': {'type': 'root_cause', 'target': 'bank_destabilization'},
    },

    {
        'name': 'deep_chain_10hop_dam',
        'type': 'root_cause',
        'context': (
            "A small earthquake in May opened micro-fractures in the Blackrock dam. "
            "The micro-fractures admitted seepage into the concrete. "
            "Seepage corroded internal rebar over six months. "
            "Corroded rebar lost tensile strength across the main spillway. "
            "Weakened spillway sections deformed under winter load. "
            "The deformation triggered an automated water-release protocol. "
            "The emergency release surged the downstream river. "
            "The river surge overwhelmed the Milltown levee. "
            "Overtopped levee flooded eight neighborhoods. "
            "Flood waters contaminated the regional water supply."
        ),
        'question': "What was the root geological cause of the water contamination?",
        'ground_truth': 'earthquake',
        'plain_fail_keyword': 'overtopped levee',
        'graph': {
            'nodes': _chain(
                ['earthquake','microfracture','seepage','corrosion','tensile_loss',
                 'deformation','release_protocol','river_surge','levee_overtop',
                 'flood','water_contamination'],
                [T_EVENT]*11
            )[0],
            'edges': _chain(
                ['earthquake','microfracture','seepage','corrosion','tensile_loss',
                 'deformation','release_protocol','river_surge','levee_overtop',
                 'flood','water_contamination'],
                [T_EVENT]*11
            )[1],
        },
        'query': {'type': 'root_cause', 'target': 'water_contamination'},
    },

    # ════════════════════════════════════════════════════════════════
    # CATEGORY 2 — Interleaved multi-chain contamination
    # ════════════════════════════════════════════════════════════════

    {
        'name': 'four_chains_same_endpoint_vocab',
        'type': 'connection_check',
        'context': (
            "The tannery fire destroyed leather stockpiles. Leather prices doubled. "
            "Doubled leather prices bankrupted small shoemakers. "
            "A dockworkers' strike halted container loading. Halted loading delayed textile exports. "
            "Delayed textiles missed the trade-fair season. "
            "A bee colony collapse cut pollination. Reduced pollination shrank fruit yields. "
            "Shrunk yields drove up jam prices. "
            "A highway tunnel flood closed the main supply route. The closed tunnel forced truck reroutes. "
            "Reroutes added six hours to produce deliveries. "
            "All four disruptions were reported on Tuesday."
        ),
        'question': "Did the tannery fire directly cause the delayed textile exports?",
        'ground_truth': 'no',
        'plain_fail_keyword': 'yes',   # plain LLM stitches chains by thematic similarity
        'graph': {
            'nodes': [
                {'id': 'tannery_fire', 'type': T_EVENT, 'role': R_ROOT},
                {'id': 'leather_loss', 'type': T_STATE, 'role': R_INTER},
                {'id': 'leather_prices', 'type': T_STATE, 'role': R_INTER},
                {'id': 'shoemaker_bankruptcy', 'type': T_CONSEQUENCE, 'role': R_TERMINAL},
                {'id': 'strike', 'type': T_EVENT, 'role': R_ROOT},
                {'id': 'loading_halt', 'type': T_STATE, 'role': R_INTER},
                {'id': 'textile_delay', 'type': T_CONSEQUENCE, 'role': R_INTER},
                {'id': 'missed_trade_fair', 'type': T_CONSEQUENCE, 'role': R_TERMINAL},
                {'id': 'bee_collapse', 'type': T_EVENT, 'role': R_ROOT},
                {'id': 'pollination_loss', 'type': T_STATE, 'role': R_INTER},
                {'id': 'fruit_yields', 'type': T_STATE, 'role': R_INTER},
                {'id': 'jam_prices', 'type': T_CONSEQUENCE, 'role': R_TERMINAL},
                {'id': 'tunnel_flood', 'type': T_EVENT, 'role': R_ROOT},
                {'id': 'tunnel_closure', 'type': T_STATE, 'role': R_INTER},
                {'id': 'reroutes', 'type': T_CONSEQUENCE, 'role': R_INTER},
                {'id': 'delivery_delay', 'type': T_CONSEQUENCE, 'role': R_TERMINAL},
            ],
            'edges': [
                {'src': 'tannery_fire', 'dst': 'leather_loss', 'type': 0},
                {'src': 'leather_loss', 'dst': 'leather_prices', 'type': 0},
                {'src': 'leather_prices', 'dst': 'shoemaker_bankruptcy', 'type': 0},
                {'src': 'strike', 'dst': 'loading_halt', 'type': 0},
                {'src': 'loading_halt', 'dst': 'textile_delay', 'type': 0},
                {'src': 'textile_delay', 'dst': 'missed_trade_fair', 'type': 0},
                {'src': 'bee_collapse', 'dst': 'pollination_loss', 'type': 0},
                {'src': 'pollination_loss', 'dst': 'fruit_yields', 'type': 0},
                {'src': 'fruit_yields', 'dst': 'jam_prices', 'type': 0},
                {'src': 'tunnel_flood', 'dst': 'tunnel_closure', 'type': 0},
                {'src': 'tunnel_closure', 'dst': 'reroutes', 'type': 0},
                {'src': 'reroutes', 'dst': 'delivery_delay', 'type': 0},
            ],
        },
        'query': {'type': 'connection_check', 'src': 'tannery_fire', 'dst': 'textile_delay'},
    },

    {
        'name': 'three_chains_textile_vs_ice',
        'type': 'connection_check',
        'context': (
            "An unseasonal heat wave accelerated ice-shelf melt. The melt released methane pockets. "
            "The methane release boosted atmospheric greenhouse forcing. "
            "Boosted forcing deepened the polar vortex disruption. "
            "A supplier audit uncovered fabric defects. The audit led to recall of 40000 garments. "
            "The recall cost the retailer 12 million dollars. "
            "A compiler bug shipped with the 4.2 release. The bug caused data corruption in three firmware builds. "
            "Corrupt firmware bricked 200 industrial controllers."
        ),
        'question': "Did the heat wave cause the firmware corruption?",
        'ground_truth': 'no',
        'plain_fail_keyword': 'yes',
        'graph': {
            'nodes': [
                {'id': 'heat_wave', 'type': T_EVENT, 'role': R_ROOT},
                {'id': 'ice_melt', 'type': T_STATE, 'role': R_INTER},
                {'id': 'methane_release', 'type': T_STATE, 'role': R_INTER},
                {'id': 'ghg_forcing', 'type': T_STATE, 'role': R_INTER},
                {'id': 'vortex_disruption', 'type': T_CONSEQUENCE, 'role': R_TERMINAL},
                {'id': 'audit', 'type': T_EVENT, 'role': R_ROOT},
                {'id': 'fabric_defect', 'type': T_STATE, 'role': R_INTER},
                {'id': 'recall', 'type': T_EVENT, 'role': R_INTER},
                {'id': 'retailer_cost', 'type': T_CONSEQUENCE, 'role': R_TERMINAL},
                {'id': 'compiler_bug', 'type': T_EVENT, 'role': R_ROOT},
                {'id': 'data_corruption', 'type': T_STATE, 'role': R_INTER},
                {'id': 'firmware_corruption', 'type': T_STATE, 'role': R_INTER},
                {'id': 'bricked_controllers', 'type': T_CONSEQUENCE, 'role': R_TERMINAL},
            ],
            'edges': [
                {'src': 'heat_wave', 'dst': 'ice_melt', 'type': 0},
                {'src': 'ice_melt', 'dst': 'methane_release', 'type': 0},
                {'src': 'methane_release', 'dst': 'ghg_forcing', 'type': 0},
                {'src': 'ghg_forcing', 'dst': 'vortex_disruption', 'type': 0},
                {'src': 'audit', 'dst': 'fabric_defect', 'type': 0},
                {'src': 'fabric_defect', 'dst': 'recall', 'type': 0},
                {'src': 'recall', 'dst': 'retailer_cost', 'type': 0},
                {'src': 'compiler_bug', 'dst': 'data_corruption', 'type': 0},
                {'src': 'data_corruption', 'dst': 'firmware_corruption', 'type': 0},
                {'src': 'firmware_corruption', 'dst': 'bricked_controllers', 'type': 0},
            ],
        },
        'query': {'type': 'connection_check', 'src': 'heat_wave', 'dst': 'firmware_corruption'},
    },

    {
        'name': 'three_chains_same_vocab_bridge',
        'type': 'connection_check',
        'context': (
            "A bridge inspection in the north district flagged structural fatigue. The inspector's report was suppressed. "
            "Suppression delayed repairs for eight months. Delayed repairs led to partial bridge collapse. "
            "A separate bridge expansion project was announced in the south district. The project cleared environmental review. "
            "The review cleared wetlands for infrastructure use. The wetland decision provoked environmental protests. "
            "A third bridge in the west was renamed after a donor. The renaming ceremony was attended by the mayor. "
            "The ceremony attendance became a campaign controversy. The controversy dominated local news for a week."
        ),
        'question': "Did the north-district bridge inspection cause the west-bridge campaign controversy?",
        'ground_truth': 'no',
        'plain_fail_keyword': 'yes',
        'graph': {
            'nodes': [
                {'id': 'inspection', 'type': T_EVENT, 'role': R_ROOT},
                {'id': 'suppression', 'type': T_EVENT, 'role': R_INTER},
                {'id': 'delayed_repairs', 'type': T_STATE, 'role': R_INTER},
                {'id': 'partial_collapse', 'type': T_CONSEQUENCE, 'role': R_TERMINAL},
                {'id': 'expansion_project', 'type': T_EVENT, 'role': R_ROOT},
                {'id': 'env_review', 'type': T_EVENT, 'role': R_INTER},
                {'id': 'wetland_decision', 'type': T_EVENT, 'role': R_INTER},
                {'id': 'protests', 'type': T_CONSEQUENCE, 'role': R_TERMINAL},
                {'id': 'renaming', 'type': T_EVENT, 'role': R_ROOT},
                {'id': 'ceremony', 'type': T_EVENT, 'role': R_INTER},
                {'id': 'campaign_controversy', 'type': T_CONSEQUENCE, 'role': R_INTER},
                {'id': 'news_week', 'type': T_CONSEQUENCE, 'role': R_TERMINAL},
            ],
            'edges': [
                {'src': 'inspection', 'dst': 'suppression', 'type': 0},
                {'src': 'suppression', 'dst': 'delayed_repairs', 'type': 0},
                {'src': 'delayed_repairs', 'dst': 'partial_collapse', 'type': 0},
                {'src': 'expansion_project', 'dst': 'env_review', 'type': 0},
                {'src': 'env_review', 'dst': 'wetland_decision', 'type': 0},
                {'src': 'wetland_decision', 'dst': 'protests', 'type': 0},
                {'src': 'renaming', 'dst': 'ceremony', 'type': 0},
                {'src': 'ceremony', 'dst': 'campaign_controversy', 'type': 0},
                {'src': 'campaign_controversy', 'dst': 'news_week', 'type': 0},
            ],
        },
        'query': {'type': 'connection_check', 'src': 'inspection', 'dst': 'campaign_controversy'},
    },

    # ════════════════════════════════════════════════════════════════
    # CATEGORY 3 — Scope-nested implication (reversed-answer scope)
    # ════════════════════════════════════════════════════════════════

    {
        'name': 'scope_pediatric_vs_adult_med',
        'type': 'implication_check',
        'context': (
            "Hospital protocol requires that patients receive treatment only after diagnosis confirmation. "
            "For the adult oncology ward, diagnosis confirmation requires a biopsy and a second-opinion review. "
            "For the pediatric ward, diagnosis confirmation requires a biopsy and a pediatric-specialist review (not a second-opinion review). "
            "Within the adult oncology scope, a biopsy plus second-opinion review implies treatment authorization. "
            "Within the pediatric scope, a biopsy plus pediatric-specialist review implies treatment authorization."
        ),
        'question': (
            "In the pediatric ward, does a biopsy plus a second-opinion review "
            "imply the patient is authorized for treatment?"
        ),
        'ground_truth': 'no',
        'plain_fail_keyword': 'yes',
        'graph': {
            'nodes': [
                {'id': 'adult_onc', 'type': T_ROLE, 'role': R_SCOPE},
                {'id': 'pediatric', 'type': T_ROLE, 'role': R_SCOPE},
                {'id': 'biopsy', 'type': T_REQUIREMENT, 'role': R_INTER},
                {'id': 'second_opinion', 'type': T_REQUIREMENT, 'role': R_INTER},
                {'id': 'ped_specialist', 'type': T_REQUIREMENT, 'role': R_INTER},
                {'id': 'diagnosis_conf', 'type': T_CREDENTIAL, 'role': R_INTER},
                {'id': 'treatment_auth', 'type': T_CREDENTIAL, 'role': R_TERMINAL},
            ],
            'edges': [
                {'src': 'adult_onc', 'dst': 'diagnosis_conf', 'type': 1},
                {'src': 'pediatric', 'dst': 'diagnosis_conf', 'type': 1},
                {'src': 'diagnosis_conf', 'dst': 'biopsy', 'type': 1},
                {'src': 'diagnosis_conf', 'dst': 'second_opinion', 'type': 1, 'scope': 'adult_onc'},
                {'src': 'diagnosis_conf', 'dst': 'ped_specialist', 'type': 1, 'scope': 'pediatric'},
                {'src': 'diagnosis_conf', 'dst': 'treatment_auth', 'type': 1},
            ],
        },
        'query': {
            'type': 'implication_check',
            'premise': 'biopsy',
            'conclusion': 'second_opinion',
            'context_scope': 'pediatric',
        },
    },

    {
        'name': 'scope_civilian_vs_military_clearance',
        'type': 'implication_check',
        'context': (
            "Secure-area access requires a valid clearance. "
            "For civilian contractors, clearance requires a background check and a fingerprint scan. "
            "For military personnel, clearance requires a background check and a command-sign-off (not a fingerprint scan). "
            "Under civilian scope, background check + fingerprint scan implies access. "
            "Under military scope, background check + command sign-off implies access."
        ),
        'question': (
            "For a military-personnel employee, does a background check with a fingerprint scan "
            "imply they have been granted secure-area access?"
        ),
        'ground_truth': 'no',
        'plain_fail_keyword': 'yes',
        'graph': {
            'nodes': [
                {'id': 'civilian', 'type': T_ROLE, 'role': R_SCOPE},
                {'id': 'military', 'type': T_ROLE, 'role': R_SCOPE},
                {'id': 'bg_check', 'type': T_REQUIREMENT, 'role': R_INTER},
                {'id': 'fingerprint', 'type': T_REQUIREMENT, 'role': R_INTER},
                {'id': 'command_signoff', 'type': T_REQUIREMENT, 'role': R_INTER},
                {'id': 'clearance', 'type': T_CREDENTIAL, 'role': R_INTER},
                {'id': 'access', 'type': T_CREDENTIAL, 'role': R_TERMINAL},
            ],
            'edges': [
                {'src': 'civilian', 'dst': 'clearance', 'type': 1},
                {'src': 'military', 'dst': 'clearance', 'type': 1},
                {'src': 'clearance', 'dst': 'bg_check', 'type': 1},
                {'src': 'clearance', 'dst': 'fingerprint', 'type': 1, 'scope': 'civilian'},
                {'src': 'clearance', 'dst': 'command_signoff', 'type': 1, 'scope': 'military'},
                {'src': 'clearance', 'dst': 'access', 'type': 1},
            ],
        },
        'query': {
            'type': 'implication_check',
            'premise': 'bg_check',
            'conclusion': 'fingerprint',
            'context_scope': 'military',
        },
    },

    {
        'name': 'scope_enterprise_vs_individual_plan',
        'type': 'implication_check',
        'context': (
            "Full system access on the platform requires an active subscription. "
            "For enterprise accounts, an active subscription requires a signed MSA and an SSO integration. "
            "For individual accounts, an active subscription requires a signed Terms of Service and a 2FA setup (not an SSO integration). "
            "For enterprise scope: signed MSA + SSO integration → active subscription. "
            "For individual scope: signed ToS + 2FA setup → active subscription."
        ),
        'question': (
            "For an individual account holder, does signing the MSA and completing the SSO integration "
            "activate their subscription?"
        ),
        'ground_truth': 'no',
        'plain_fail_keyword': 'yes',
        'graph': {
            'nodes': [
                {'id': 'enterprise', 'type': T_ROLE, 'role': R_SCOPE},
                {'id': 'individual', 'type': T_ROLE, 'role': R_SCOPE},
                {'id': 'msa', 'type': T_REQUIREMENT, 'role': R_INTER},
                {'id': 'sso', 'type': T_REQUIREMENT, 'role': R_INTER},
                {'id': 'tos', 'type': T_REQUIREMENT, 'role': R_INTER},
                {'id': 'twofa', 'type': T_REQUIREMENT, 'role': R_INTER},
                {'id': 'subscription', 'type': T_CREDENTIAL, 'role': R_INTER},
                {'id': 'access', 'type': T_CREDENTIAL, 'role': R_TERMINAL},
            ],
            'edges': [
                {'src': 'enterprise', 'dst': 'subscription', 'type': 1},
                {'src': 'individual', 'dst': 'subscription', 'type': 1},
                {'src': 'subscription', 'dst': 'msa', 'type': 1, 'scope': 'enterprise'},
                {'src': 'subscription', 'dst': 'sso', 'type': 1, 'scope': 'enterprise'},
                {'src': 'subscription', 'dst': 'tos', 'type': 1, 'scope': 'individual'},
                {'src': 'subscription', 'dst': 'twofa', 'type': 1, 'scope': 'individual'},
                {'src': 'subscription', 'dst': 'access', 'type': 1},
            ],
        },
        'query': {
            'type': 'implication_check',
            'premise': 'msa',
            'conclusion': 'sso',
            'context_scope': 'individual',
        },
    },

    # ════════════════════════════════════════════════════════════════
    # CATEGORY 4 — Temporal reverse-order context
    # ════════════════════════════════════════════════════════════════

    {
        'name': 'reverse_order_blackout',
        'type': 'root_cause',
        'context': (
            "The city hospital activated generator power last night. "
            "Emergency generator power was activated because grid power failed at the hospital district. "
            "The grid power failure was triggered by a substation automatic shutdown. "
            "The substation shutdown was initiated because of a voltage surge on the primary feed. "
            "The voltage surge was caused by a squirrel making contact with the feed transformer."
        ),
        'question': "What was the actual root cause of the hospital generator activation?",
        'ground_truth': 'squirrel',
        'plain_fail_keyword': 'grid power failed',
        'graph': {
            'nodes': _chain(
                ['squirrel','voltage_surge','substation_shutdown','grid_failure',
                 'generator_activation'],
                [T_EVENT]*5
            )[0],
            'edges': _chain(
                ['squirrel','voltage_surge','substation_shutdown','grid_failure',
                 'generator_activation'],
                [T_EVENT]*5
            )[1],
        },
        'query': {'type': 'root_cause', 'target': 'generator_activation'},
    },

    {
        'name': 'reverse_order_recall',
        'type': 'root_cause',
        'context': (
            "The automotive recall issued Tuesday affected 120000 vehicles. "
            "The recall was triggered by a federal safety advisory. "
            "The safety advisory followed an investigation into brake failures. "
            "The investigation opened after multiple rear-end collisions were reported. "
            "The collisions were traced to a pad friction degradation. "
            "The friction degradation was caused by a supplier material substitution earlier that year."
        ),
        'question': "What ultimately caused the automotive recall?",
        'ground_truth': 'supplier material substitution',
        'plain_fail_keyword': 'safety advisory',
        'graph': {
            'nodes': _chain(
                ['material_substitution','friction_degradation','collisions','investigation',
                 'safety_advisory','recall'],
                [T_EVENT]*6
            )[0],
            'edges': _chain(
                ['material_substitution','friction_degradation','collisions','investigation',
                 'safety_advisory','recall'],
                [T_EVENT]*6
            )[1],
        },
        'query': {'type': 'root_cause', 'target': 'recall'},
    },

    # ════════════════════════════════════════════════════════════════
    # CATEGORY 5 — Lexical decoy root (a plausible-but-wrong root node
    #    shares vocabulary with the question — plain LLM anchors on it)
    # ════════════════════════════════════════════════════════════════

    {
        'name': 'lexical_decoy_airport_delay',
        'type': 'root_cause',
        'context': (
            "A radar outage disrupted Terminal C operations on Thursday. "
            "The outage was handled by manual procedures within 20 minutes. "
            "A fuel-truck jackknife earlier that morning blocked the taxiway for two hours. "
            "The taxiway block delayed 18 departures. The delayed departures rippled into connecting hubs. "
            "Ripple effects caused passenger stranding at six airports by end of day."
        ),
        'question': "What was the root cause of the passenger stranding?",
        'ground_truth': 'fuel-truck jackknife',
        # Plain LLM may anchor on "radar outage" because it's mentioned first and associated with airport disruption
        'plain_fail_keyword': 'radar',
        'graph': {
            'nodes': [
                {'id': 'radar_outage', 'type': T_EVENT, 'role': R_ROOT},
                {'id': 'manual_procedures', 'type': T_STATE, 'role': R_TERMINAL},
                {'id': 'fuel_jackknife', 'type': T_EVENT, 'role': R_ROOT},
                {'id': 'taxiway_block', 'type': T_STATE, 'role': R_INTER},
                {'id': 'departure_delay', 'type': T_CONSEQUENCE, 'role': R_INTER},
                {'id': 'ripple', 'type': T_CONSEQUENCE, 'role': R_INTER},
                {'id': 'stranding', 'type': T_CONSEQUENCE, 'role': R_TERMINAL},
            ],
            'edges': [
                {'src': 'radar_outage', 'dst': 'manual_procedures', 'type': 0},
                {'src': 'fuel_jackknife', 'dst': 'taxiway_block', 'type': 0},
                {'src': 'taxiway_block', 'dst': 'departure_delay', 'type': 0},
                {'src': 'departure_delay', 'dst': 'ripple', 'type': 0},
                {'src': 'ripple', 'dst': 'stranding', 'type': 0},
            ],
        },
        'query': {'type': 'root_cause', 'target': 'stranding'},
    },

    {
        'name': 'lexical_decoy_data_breach',
        'type': 'root_cause',
        'context': (
            "A phishing campaign targeted the finance department's email last month. Training was reissued and the campaign was contained. "
            "Independently, a third-party analytics vendor was integrated with the CRM in March. "
            "The vendor's SDK quietly logged session tokens. "
            "Logged tokens were exfiltrated when the vendor's dashboard was hacked. "
            "Exfiltrated tokens allowed attackers into our CRM. "
            "CRM access exposed 40000 customer records."
        ),
        'question': "What was the root cause of the customer record exposure?",
        'ground_truth': 'third-party analytics vendor',
        # Plain LLM may anchor on "phishing campaign" because it's mentioned first and associated with breaches
        'plain_fail_keyword': 'phishing',
        'graph': {
            'nodes': [
                {'id': 'phishing', 'type': T_EVENT, 'role': R_ROOT},
                {'id': 'training', 'type': T_STATE, 'role': R_TERMINAL},
                {'id': 'vendor_integration', 'type': T_EVENT, 'role': R_ROOT},
                {'id': 'sdk_logging', 'type': T_STATE, 'role': R_INTER},
                {'id': 'vendor_hack', 'type': T_EVENT, 'role': R_INTER},
                {'id': 'token_exfil', 'type': T_CONSEQUENCE, 'role': R_INTER},
                {'id': 'crm_access', 'type': T_CONSEQUENCE, 'role': R_INTER},
                {'id': 'record_exposure', 'type': T_CONSEQUENCE, 'role': R_TERMINAL},
            ],
            'edges': [
                {'src': 'phishing', 'dst': 'training', 'type': 0},
                {'src': 'vendor_integration', 'dst': 'sdk_logging', 'type': 0},
                {'src': 'sdk_logging', 'dst': 'vendor_hack', 'type': 0},
                {'src': 'vendor_hack', 'dst': 'token_exfil', 'type': 0},
                {'src': 'token_exfil', 'dst': 'crm_access', 'type': 0},
                {'src': 'crm_access', 'dst': 'record_exposure', 'type': 0},
            ],
        },
        'query': {'type': 'root_cause', 'target': 'record_exposure'},
    },

    # ════════════════════════════════════════════════════════════════
    # CATEGORY 6 — Branching / merging (multiple roots reach effect)
    # ════════════════════════════════════════════════════════════════

    {
        'name': 'merging_diagnostic',
        'type': 'connection_check',
        'context': (
            "A transformer fire knocked out power to Zone 3. Zone-3 power loss caused refrigerator thaw. "
            "Refrigerator thaw spoiled 10000 liters of dairy. "
            "Separately, a quality-lab glitch reported elevated bacterial counts. Elevated counts triggered a precautionary recall. "
            "The recall pulled 10000 liters of dairy from stores. Both events reported the same 10000-liter loss on Tuesday."
        ),
        'question': "Did the transformer fire directly cause the precautionary recall?",
        'ground_truth': 'no',
        'plain_fail_keyword': 'yes',
        'graph': {
            'nodes': [
                {'id': 'transformer_fire', 'type': T_EVENT, 'role': R_ROOT},
                {'id': 'zone3_power_loss', 'type': T_STATE, 'role': R_INTER},
                {'id': 'fridge_thaw', 'type': T_STATE, 'role': R_INTER},
                {'id': 'dairy_spoiled_A', 'type': T_CONSEQUENCE, 'role': R_TERMINAL},
                {'id': 'lab_glitch', 'type': T_EVENT, 'role': R_ROOT},
                {'id': 'bacterial_report', 'type': T_STATE, 'role': R_INTER},
                {'id': 'recall', 'type': T_EVENT, 'role': R_INTER},
                {'id': 'dairy_pulled_B', 'type': T_CONSEQUENCE, 'role': R_TERMINAL},
            ],
            'edges': [
                {'src': 'transformer_fire', 'dst': 'zone3_power_loss', 'type': 0},
                {'src': 'zone3_power_loss', 'dst': 'fridge_thaw', 'type': 0},
                {'src': 'fridge_thaw', 'dst': 'dairy_spoiled_A', 'type': 0},
                {'src': 'lab_glitch', 'dst': 'bacterial_report', 'type': 0},
                {'src': 'bacterial_report', 'dst': 'recall', 'type': 0},
                {'src': 'recall', 'dst': 'dairy_pulled_B', 'type': 0},
            ],
        },
        'query': {'type': 'connection_check', 'src': 'transformer_fire', 'dst': 'recall'},
    },

    # ════════════════════════════════════════════════════════════════
    # CATEGORY 7 — Long-distance scope-gated implication
    # ════════════════════════════════════════════════════════════════

    {
        'name': 'scope_long_distance_audit',
        'type': 'implication_check',
        'context': (
            "Final financial sign-off requires a completed audit package. "
            "A completed audit package requires auditor attestation plus management representation. "
            "Auditor attestation requires completed substantive testing and completed control testing. "
            "For public companies (scope: public), substantive testing additionally requires an ICFR opinion. "
            "For private companies (scope: private), substantive testing requires only the standard sample review (no ICFR opinion)."
        ),
        'question': (
            "For a private company, does completing substantive testing require an ICFR opinion?"
        ),
        'ground_truth': 'no',
        'plain_fail_keyword': 'yes',
        'graph': {
            'nodes': [
                {'id': 'public', 'type': T_ROLE, 'role': R_SCOPE},
                {'id': 'private', 'type': T_ROLE, 'role': R_SCOPE},
                {'id': 'financial_signoff', 'type': T_CREDENTIAL, 'role': R_TERMINAL},
                {'id': 'audit_package', 'type': T_CREDENTIAL, 'role': R_INTER},
                {'id': 'attestation', 'type': T_CREDENTIAL, 'role': R_INTER},
                {'id': 'mgmt_rep', 'type': T_REQUIREMENT, 'role': R_INTER},
                {'id': 'substantive_testing', 'type': T_REQUIREMENT, 'role': R_INTER},
                {'id': 'control_testing', 'type': T_REQUIREMENT, 'role': R_INTER},
                {'id': 'icfr_opinion', 'type': T_PREREQUISITE, 'role': R_INTER},
                {'id': 'sample_review', 'type': T_PREREQUISITE, 'role': R_INTER},
            ],
            'edges': [
                {'src': 'public', 'dst': 'audit_package', 'type': 1},
                {'src': 'private', 'dst': 'audit_package', 'type': 1},
                {'src': 'audit_package', 'dst': 'financial_signoff', 'type': 1},
                {'src': 'audit_package', 'dst': 'attestation', 'type': 1},
                {'src': 'audit_package', 'dst': 'mgmt_rep', 'type': 1},
                {'src': 'attestation', 'dst': 'substantive_testing', 'type': 1},
                {'src': 'attestation', 'dst': 'control_testing', 'type': 1},
                {'src': 'substantive_testing', 'dst': 'icfr_opinion', 'type': 1, 'scope': 'public'},
                {'src': 'substantive_testing', 'dst': 'sample_review', 'type': 1, 'scope': 'private'},
            ],
        },
        'query': {
            'type': 'implication_check',
            'premise': 'substantive_testing',
            'conclusion': 'icfr_opinion',
            'context_scope': 'private',
        },
    },
]


if __name__ == '__main__':
    print(f"TESTS_HARD: {len(TESTS_HARD)} tests")
    for t in TESTS_HARD:
        print(f"  - {t['name']:<40}  type={t['type']}  gt={t['ground_truth']}")
