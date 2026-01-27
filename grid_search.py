import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import sys
from matplotlib import pyplot as plt
from google.maps import places_v1
import json
import time
from google.protobuf.json_format import MessageToDict
os.chdir(sys.path[0])

load_dotenv(override=True)
client_id = os.getenv('ID')
client = places_v1.PlacesClient(client_options={"api_key": client_id})

metadata = (("x-goog-fieldmask", "places.displayName,places.id,places.location,places.primaryType,places.types,places.primaryTypeDisplayName,places.priceLevel,places.priceRange,places.rating,places.userRatingCount"),)

location = (32.709653, -117.171488)
radius = 100  # meters

# --- Load existing data if present ---
existing_places = []
seen_ids = set()

output_file = "places_grid_search.json"

if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        existing_places = json.load(f)

    # Seed seen_ids from previously saved places
    for p in existing_places:
        if "id" in p:
            seen_ids.add(p["id"])

print(f"Loaded {len(existing_places)} existing places")

places = []
for i in range(60):
    for j in range(20, 50):       
        response = client.search_nearby({
            'location_restriction': {
                'circle': {
                    'center': {'latitude': location[0]+0.0009*i, 'longitude': location[1]+0.0011*j},
                    'radius': radius
                }
            },

            'included_primary_types': {'restaurant'},
            },
            metadata=metadata
            )
        for place in response.places:
            if place.id not in seen_ids:
                seen_ids.add(place.id)  # mark as seen
                places.extend([place])
                
        time.sleep(0.11)  # ~9 req/sec, stay under request quota

# Convert only NEW places to dicts
new_json_places = [
    MessageToDict(
        place._pb,
        preserving_proto_field_name=True,
        use_integers_for_enums=True,
    )
    for place in places
]

# Combine old + new
combined_places = existing_places + new_json_places

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(combined_places, f, indent=2, ensure_ascii=False)

print(f"New places fetched this run: {len(new_json_places)}")
print(f"Total unique places stored: {len(combined_places)}")