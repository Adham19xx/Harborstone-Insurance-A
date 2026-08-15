"""Real Harborstone Marine Insurance request fixtures for comprehensive planning evaluation.

Contains 15 diverse, realistic insurance scenarios to evaluate:
- Decomposition-first vs. Dynamic decomposition
- Plan-and-Solve (PS) vs. Tree of Thoughts (ToT) vs. LATS
- Self-Refine vs. Reflexion
- Grounded vs. Ungrounded Environment critiques
"""

from typing import Any, Dict, List

REAL_REQUESTS: List[Dict[str, Any]] = [
    # 1. Eligible Standard Boat Addition (Favors Decomposition-First)
    {
        "request_id": "marine-policy-update-eligible-vessel",
        "customer_id": 1,
        "category": "decomposition_first",
        "new_vessel": {
            "vessel_name": "Bay Runner",
            "vessel_type": "Boat",
            "manufacturer": "Boston Whaler",
            "model": "Outrage 280",
            "year_built": 2024,
            "value": 180000.00,
        },
        "current_premium": 2100.00,
        "text": (
            "I am a Harborstone Insurance customer and recently purchased a 2024 Boston Whaler. "
            "Please review my existing marine policy, verify eligibility, determine coverage impact, "
            "estimate the premium increase, and outline required documents."
        ),
    },

    # 2. Ineligible Over-Age Vessel (Favors Dynamic Decomposition Early Divergence)
    {
        "request_id": "marine-policy-update-ineligible-vessel",
        "customer_id": 1,
        "category": "dynamic_decomposition",
        "new_vessel": {
            "vessel_name": "Gulf Star",
            "vessel_type": "Yacht",
            "manufacturer": "Sunseeker",
            "model": "Predator 55",
            "year_built": 2000,  # 26 years old -> exceeds 20 year limit
            "value": 500000.00,
        },
        "current_premium": 2100.00,
        "text": (
            "I am a Harborstone Insurance customer and purchased a classic Sunseeker Yacht. "
            "Please review my policy, verify vessel eligibility, calculate premium changes, "
            "and finalize the addition."
        ),
    },

    # 3. High-Value Luxury Yacht Requiring Surveyor Appraisal (LATS + Grounded)
    {
        "request_id": "marine-yacht-luxury-survey",
        "customer_id": 2,
        "category": "lats",
        "new_vessel": {
            "vessel_name": "Oceanic Sovereign",
            "vessel_type": "Yacht",
            "manufacturer": "Azimut",
            "model": "Grande 27M",
            "year_built": 2023,
            "value": 1200000.00,  # >= $500k -> requires independent surveyor appraisal
        },
        "current_premium": 4500.00,
        "text": (
            "I need to endorse my high-value luxury Azimut yacht onto my existing commercial hull policy. "
            "Provide a compliant endorsement structure, ensure all survey regulations are met, and calculate premium."
        ),
    },

    # 4. Multi-Option Deductible & Coverage Tradeoff (Tree of Thoughts - BFS)
    {
        "request_id": "marine-multi-option-deductible-optimization",
        "customer_id": 3,
        "category": "tree_of_thoughts",
        "new_vessel": {
            "vessel_name": "Sea Explorer",
            "vessel_type": "Yacht",
            "manufacturer": "Princess",
            "model": "F55",
            "year_built": 2022,
            "value": 650000.00,
        },
        "current_premium": 3200.00,
        "text": (
            "I want to explore different deductible tiers (low, medium, high) and navigation limits "
            "for my Princess yacht to find the best risk vs premium optimization."
        ),
    },

    # 5. Multi-Vessel Portfolio Risk Prioritization (Tree of Thoughts - DFS)
    {
        "request_id": "marine-portfolio-risk-ranking",
        "customer_id": 4,
        "category": "tree_of_thoughts",
        "new_vessel": {
            "vessel_name": "Fleet Tender Alpha",
            "vessel_type": "Boat",
            "manufacturer": "Sea Ray",
            "model": "SDX 290",
            "year_built": 2021,
            "value": 140000.00,
        },
        "current_premium": 1800.00,
        "text": (
            "We are adding multiple tender boats to our marine account. Rank the risk profile and "
            "prioritize endorsement order to minimize upfront cash outlay while maintaining full liability."
        ),
    },

    # 6. Actuarial Tiered Formula Premium Calculation (Plan-and-Solve)
    {
        "request_id": "marine-actuarial-tiered-pricing",
        "customer_id": 5,
        "category": "plan_and_solve",
        "new_vessel": {
            "vessel_name": "Coastal Cruiser",
            "vessel_type": "Boat",
            "manufacturer": "Grady-White",
            "model": "Canyon 336",
            "year_built": 2018,
            "value": 250000.00,
        },
        "current_premium": 1950.00,
        "deductible": 12500.00,
        "text": (
            "Calculate step-by-step the exact premium adjustment for our 2018 Grady-White, including "
            "base rate (1.0%), age surcharge evaluation, and 5% high-deductible loyalty discount."
        ),
    },

    # 7. Deductible Discount & Fee Schedule Computation (Plan-and-Solve)
    {
        "request_id": "marine-deductible-rate-math",
        "customer_id": 6,
        "category": "plan_and_solve",
        "new_vessel": {
            "vessel_name": "Blue Horizon",
            "vessel_type": "Yacht",
            "manufacturer": "Beneteau",
            "model": "Oceanis 46.1",
            "year_built": 2024,
            "value": 420000.00,
        },
        "current_premium": 2800.00,
        "deductible": 21000.00,
        "text": (
            "Execute the sequential actuarial formula for Beneteau Yacht addition: Yacht rate 1.5%, "
            "deductible discount check, administrative policy amendment fee, and new annual total."
        ),
    },

    # 8. Customer Endorsement Notification Synthesis (Self-Refine)
    {
        "request_id": "marine-endorsement-customer-notice",
        "customer_id": 7,
        "category": "self_refine",
        "new_vessel": {
            "vessel_name": "Island Hopper",
            "vessel_type": "Boat",
            "manufacturer": "Cobalt",
            "model": "R30",
            "year_built": 2023,
            "value": 220000.00,
        },
        "current_premium": 1500.00,
        "total_new_premium": 3700.00,
        "text": (
            "Draft and refine a comprehensive, professional policyholder notice confirming the vessel addition, "
            "detailing the required verification documents, premium breakdown, and underwriting binding terms."
        ),
    },

    # 9. Multi-Constraint Policy Endorsement with Deductible Conflict (Reflexion)
    {
        "request_id": "marine-complex-endorsement-conflict",
        "customer_id": 8,
        "category": "reflexion",
        "new_vessel": {
            "vessel_name": "Apex Voyager",
            "vessel_type": "Yacht",
            "manufacturer": "Ferretti",
            "model": "720",
            "year_built": 2024,
            "value": 850000.00,  # Luxury yacht needing surveyor appraisal & compliant deductible
        },
        "current_premium": 5200.00,
        "text": (
            "Endorse a $850k Ferretti Yacht onto policy #104. The plan must satisfy all strict underwriting rules, "
            "minimum deductible thresholds, mandatory independent surveyor appraisal, and rate tables."
        ),
    },

    # 10. Grounded vs Ungrounded Contrast Failure Case
    {
        "request_id": "marine-ungrounded-hallucination-contrast",
        "customer_id": 9,
        "category": "grounded_contrast",
        "new_vessel": {
            "vessel_name": "Heritage Mariner",
            "vessel_type": "Yacht",
            "manufacturer": "Custom Wooden Hull",
            "model": "Classic 60",
            "year_built": 1995,  # 31 years old (exceeds 20 yr rule)
            "value": 750000.00,  # >= $500k but lacks surveyor appraisal
        },
        "current_premium": 3000.00,
        "proposed_premium": 9500.00,  # Inaccurate rate
        "deductible": 200.00,  # Below $500 min
        "text": (
            "Review proposed addition for 1995 Custom Wooden Yacht. Test whether the ungrounded critic "
            "naively accepts the plan while the grounded environment flags the age limit, missing survey, and rate violation."
        ),
    },

    # 11. Commercial Charter Yacht Addition
    {
        "request_id": "marine-commercial-charter-yacht",
        "customer_id": 10,
        "category": "lats",
        "new_vessel": {
            "vessel_name": "Aegean Wind",
            "vessel_type": "Yacht",
            "manufacturer": "Lagoon",
            "model": "Fifty 5",
            "year_built": 2022,
            "value": 980000.00,
        },
        "current_premium": 6000.00,
        "text": (
            "Add high-capacity catamaran yacht for commercial charter operations. Validate survey documents, "
            "underwriting premium rate, and maximum deductible limitations."
        ),
    },

    # 12. Speed Boat Addition with High Engine Rating
    {
        "request_id": "marine-fleet-addition-speed-boat",
        "customer_id": 11,
        "category": "decomposition_first",
        "new_vessel": {
            "vessel_name": "Velocity Pro",
            "vessel_type": "Boat",
            "manufacturer": "Cigarette",
            "model": "38 Top Gun",
            "year_built": 2023,
            "value": 350000.00,
        },
        "current_premium": 2400.00,
        "text": (
            "Add 2023 high-performance speed boat to policy. Query customer active coverage, calculate 1.0% "
            "base premium addition, and generate documentation checklist."
        ),
    },

    # 13. Vintage Restored Boat (Ineligible Age Rule)
    {
        "request_id": "marine-vintage-restored-boat",
        "customer_id": 12,
        "category": "dynamic_decomposition",
        "new_vessel": {
            "vessel_name": "Old Glory",
            "vessel_type": "Boat",
            "manufacturer": "Chris-Craft",
            "model": "Coronet 21",
            "year_built": 1998,  # Age 28 years -> Ineligible
            "value": 95000.00,
        },
        "current_premium": 1100.00,
        "text": (
            "Requesting addition of restored 1998 Chris-Craft. System must observe ineligibility and "
            "route to vintage specialty requirements instead of calculating standard premium."
        ),
    },

    # 14. Offshore Cruiser Navigation Endorsement
    {
        "request_id": "marine-offshore-cruiser-endorsement",
        "customer_id": 13,
        "category": "tree_of_thoughts",
        "new_vessel": {
            "vessel_name": "Nordic Star",
            "vessel_type": "Boat",
            "manufacturer": "Nord Star",
            "model": "31 Patrol",
            "year_built": 2021,
            "value": 290000.00,
        },
        "current_premium": 2200.00,
        "text": (
            "Evaluate navigational limit options (Inland vs Coastal vs Open Ocean) for Nord Star boat "
            "to balance premium impact with navigation freedom."
        ),
    },

    # 15. Vessel Replacement with Yacht Upgrade
    {
        "request_id": "marine-vessel-upgrade-replacement",
        "customer_id": 14,
        "category": "reflexion",
        "new_vessel": {
            "vessel_name": "Starlight Empress",
            "vessel_type": "Yacht",
            "manufacturer": "Sunseeker",
            "model": "Manhattan 68",
            "year_built": 2024,
            "value": 720000.00,
        },
        "current_premium": 3100.00,
        "text": (
            "Policyholder replacing existing vessel with new $720k Sunseeker Yacht. Structure endorsement "
            "with verified surveyor report, correct 1.5% yacht rate, and compliant deductible."
        ),
    },
]

INELIGIBLE_REQUEST = REAL_REQUESTS[1]
ELIGIBLE_REQUEST = REAL_REQUESTS[0]
