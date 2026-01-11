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

places = []
seen_ids = set()
for i in range(50):
    for j in range(10):
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

json_places = [
    MessageToDict(
        place._pb,
        preserving_proto_field_name=True,
        use_integers_for_enums=True,
    )
    for place in places
]

with open("places_grid_search.json", "w", encoding="utf-8") as f:
    json.dump(json_places, f, indent=2, ensure_ascii=False)