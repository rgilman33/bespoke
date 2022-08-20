import subprocess
import xmltodict # using this instead of overpy to parse OSM data manually ourselves
import numpy as np

OSM_DB_PATH = "/media/beans/ssd/osm/db"

def get_big_map():
    # NOTE currently just returns map for silverton area

    # We'll call this once in the beginning. Just keep loaded in memory. When venturing further out, 
    # we'll need to refresh this every so often.
    # total process of fetching 10 - 20 km radius in each direction will be a couple hundred milliseconds
    # in production version, this will probably be running every minute or so. We have time to keep a healthy buffer

    """far_radius = 40_000 #10_000
    lat, lon = 45.023867, -122.741862
    bbox_angle = np.degrees(far_radius / EARTH_RADIUS)
    bbox_str = f'{str(lat - bbox_angle)},{str(lon - bbox_angle)},{str(lat + bbox_angle)},{str(lon + bbox_angle)}'"""

    silverton_area_bbox_str = f'{str(44.85017681566702)},{-122.83272781942897},{45.07848012852752},{-122.49831789424171}'
    bbox_str = silverton_area_bbox_str
    q = """
        way(""" + bbox_str + """)
          [highway]
          [highway!~"^(footway|path|corridor|bridleway|steps|cycleway|construction|bus_guideway|escape|service|track)$"];
        (._;>;);
        out;
        """

    # grabbing osm data from our local db. Could also grab this from online, if wanted eg big dump of certain area, could then
    # store that offline and not need to deal w local db at all
    completion = subprocess.run(["/home/beans/osm-3s_v0.7.56.9/bin/osm3s_query", f"--db-dir={OSM_DB_PATH}", f'--request={q}'], 
                                                                    check=True, capture_output=True)
    
    # Manual parsing of osm data, which is just xml after all
    osm_db_out = xmltodict.parse(completion.stdout)

    # Rearranging our data in a flat way so we can filter quickly based on node distance from ego, THEN group by way
    # this cell and the next outputs flat tabular style data w three properties: way_id, lat, lon
    # prep nodes_dict for faster lookup below
    ways = osm_db_out['osm']['way']
    nodes = osm_db_out['osm']['node']
    nodes_dict = {}
    for n in nodes:
        nodes_dict[n['@id']] = (float(n["@lat"]), float(n["@lon"]))
    # create the flattened data, nodes are in order bc we're arranging using the 'nd' attribute
    way_ids, lats, lons = [], [], []
    for way in (ways if type(ways)==list else [ways]):
        way_id = int(way["@id"])
        for nd in way['nd']:
            nd_id = nd['@ref']
            node = nodes_dict[nd_id]
            way_ids.append(way_id)
            lats.append(node[0])
            lons.append(node[1])
    lats, lons, way_ids = np.array(lats), np.array(lons), np.array(way_ids)
    
    return lats, lons, way_ids

    