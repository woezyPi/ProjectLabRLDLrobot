from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OUTPUT = ROOT / "worlds/FederatedRL_v2.wbt"

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "cyberbotics/webots/R2025a/projects"
)

header = f'''#VRML_SIM R2025a utf8

EXTERNPROTO "{BASE_URL}/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "{BASE_URL}/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "{BASE_URL}/objects/floors/protos/CircleArena.proto"
EXTERNPROTO "{BASE_URL}/objects/factory/containers/protos/WoodenBox.proto"
EXTERNPROTO "{BASE_URL}/robots/robotis/turtlebot/protos/TurtleBot3Burger.proto"

WorldInfo {{
  basicTimeStep 64
}}

Viewpoint {{
  orientation 0 1 0 1.57079632679
  position 0 0 30
}}

TexturedBackground {{
}}

TexturedBackgroundLight {{
}}
'''

# Configuration des arènes

arenas = [
    {
        "id": 1,
        "center": (-7, 0),
        "fixed": [
            (-1.2, 0.8),
            (1.2, 0.8),
            (0.0, -1.5)
        ],
        "mobile": [
            (1.2, -0.8)
        ]
    },
    {
        "id": 2,
        "center": (0, 0),
        "fixed": [
            (-1.2, 0.8),
            (1.2, 0.8),
            (0.0, -1.5),
            (-1.5, -0.8),
            (0.0, 1.8)
        ],
        "mobile": [
            (1.2, -0.8),
            (-0.7, 1.5)
        ]
    },
    {
        "id": 3,
        "center": (7, 0),
        "fixed": [
            (-1.2, 0.8),
            (1.2, 0.8),
            (0.0, -1.5),
            (-1.5, -0.8),
            (0.0, 1.8),
            (1.8, 0),
            (-1.8, 0),
            (0.8, -1.9)
        ],
        "mobile": [
            (1.2, -0.8),
            (-0.7, 1.5),
            (-0.8, -1.6)
        ]
    }
]

world = [header]

mobile_id = 1

for arena in arenas:

    arena_id = arena["id"]
    cx, cy = arena["center"]

    # Arène
    world.append(f'''
DEF ARENA_{arena_id} CircleArena {{
  translation {cx} {cy} 0
  name "arena_{arena_id}"
  radius 3
  wallHeight 0.4
}}
''')

    # Obstacles fixes
    for i, (x, y) in enumerate(arena["fixed"], 1):

        world.append(f'''
DEF STATIC_{arena_id}_{i} WoodenBox {{
  translation {cx + x} {cy + y} 0.15
  name "static_{arena_id}_{i}"
  size 0.3 0.3 0.3
}}
''')

    # Obstacles mobiles
    for x, y in arena["mobile"]:

        world.append(f'''
DEF OBSTACLE_{mobile_id} WoodenBox {{
  translation {cx + x} {cy + y} 0.15
  name "mobile_{mobile_id}"
  size 0.3 0.3 0.3
}}
''')

        mobile_id += 1

    # Agent
    world.append(f'''
DEF ROBOT_{arena_id} TurtleBot3Burger {{
  translation {cx} {cy} 0
  rotation 0 0 1 0
  name "robot_{arena_id}"
  controller "frl_agent"
  supervisor FALSE
  synchronization TRUE
}}
''')

# Supervisor global
world.append('''
DEF ENV_SUPERVISOR Robot {
  name "superviser"
  controller "superviser"
  supervisor TRUE
  synchronization TRUE
}
''')

OUTPUT.write_text(
    "\n".join(world),
    encoding="utf-8"
)

print(f"Monde généré : {OUTPUT}")
print(f"Arènes : {len(arenas)}")
print(f"Robots : {len(arenas)}")
print(f"Obstacles mobiles : {mobile_id - 1}")