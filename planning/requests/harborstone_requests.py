"""Real Harborstone request fixtures based on the existing Marine schema.

The existing Harborstone database models vessels, not automobiles, so the request type
is the marine equivalent of the selected policy-update request: add a newly purchased
vessel to an existing policy. No new database schema is introduced.
"""

INELIGIBLE_REQUEST = {
    "request_id": "marine-policy-update-ineligible-vessel",
    "customer_id": 1,
    "new_vessel": {
        "vessel_name": "Gulf Star",
        "vessel_type": "Yacht",
        "manufacturer": "Sunseeker",
        "model": "Predator 55",
        "year_built": 2000,
        "value": 500000.00,
    },
    "text": (
        "I am a Harborstone Insurance customer and recently purchased a new vessel. "
        "Please review my existing marine policy, check whether the new vessel can be "
        "added, determine coverage impact, estimate the premium change, identify required "
        "information, and give me the final next steps."
    ),
}

ELIGIBLE_REQUEST = {
    **INELIGIBLE_REQUEST,
    "request_id": "marine-policy-update-eligible-vessel",
    "new_vessel": {**INELIGIBLE_REQUEST["new_vessel"], "year_built": 2024},
}

REAL_REQUESTS = [INELIGIBLE_REQUEST, ELIGIBLE_REQUEST]
