# Define room
room = Room(length=5.0, width=5.0)

# Derived 2D wall segments (for reference)
walls = {
    "L": ((0.0, 0.0), (0.0, room.width)),
    "R": ((room.length, 0.0), (room.length, room.width)),
    "B": ((0.0, 0.0), (room.length, 0.0)),
    "T": ((0.0, room.width), (room.length, room.width))
}

# -----------------------
# Var declarations
# -----------------------

# Sleeping area (bed against L wall, vertically centered)
bed_center_y = Var(2.5)

# Nightstands: shared symmetric parameters
ns_clearance = Var(0.2)
ns_alignment = Var("backboard")
ns_align_angle = Var(0.0)

# Work nook internals
chair_front_gap = Var(0.3)
desk_lamp_gap = Var(0.2)
desk_member_align = Var(0.0)

# Reading nook internals
reading_side_gap = Var(0.2)
reading_member_align = Var(0.0)

# Global placements
dresser_center_y = Var(2.5)   # sideboard centered on right wall
wardrobe_center_y = Var(1.6)  # wardrobe vertical position on right wall (separated from sideboard)
mirror_top_x = Var(1.0)       # mirror placed on top wall left section

# -----------------------
# Cluster: Sleeping Area
# -----------------------
with solver.cluster(cluster_id="sleeping_area", anchor=bed_0, members=[bed_0, nightstand_0, nightstand_1]) as sleeping_area:
    # Place nightstands symmetrically with backboard alignment; align rotations with bed
    solver.left_of(source=nightstand_0, target=bed_0, clearance=ns_clearance, alignment=ns_alignment)
    solver.align(source=nightstand_0, target=bed_0, angle=ns_align_angle)

    solver.right_of(source=nightstand_1, target=bed_0, clearance=ns_clearance, alignment=ns_alignment)
    solver.align(source=nightstand_1, target=bed_0, angle=ns_align_angle)

    # Local scaffold (anchor at origin, facing +Y)
    # Bed (2.3 x 2.6), rz=0 → NO SWAP
    bed_0.footprint = Footprint(rz=0, lx=2.3, ly=2.6)
    bed_0.pose = Pose(x=0.0, y=0.0, rz=0.0)
    bed_0.bounds = Bounds(x_min=-1.15, x_max=1.15, y_min=-1.3, y_max=1.3)

    # Nightstand 0 (left): (0.5 x 0.4), rz=0 → NO SWAP
    nightstand_0.footprint = Footprint(rz=0, lx=0.5, ly=0.4)
    nightstand_0.pose = Pose(x=-1.6, y=-1.1, rz=0.0)
    nightstand_0.bounds = Bounds(x_min=-1.85, x_max=-1.35, y_min=-1.3, y_max=-0.9)

    # Nightstand 1 (right): (0.5 x 0.4), rz=0 → NO SWAP
    nightstand_1.footprint = Footprint(rz=0, lx=0.5, ly=0.4)
    nightstand_1.pose = Pose(x=1.6, y=-1.1, rz=0.0)
    nightstand_1.bounds = Bounds(x_min=1.35, x_max=1.85, y_min=-1.3, y_max=-0.9)

# Cluster AABB (local)
sleeping_area.aabb = AABB(lx=3.7, ly=2.6)

# -----------------------
# Cluster: Work Nook (desk + chair + floor lamp)
# -----------------------
with solver.cluster(cluster_id="work_nook", anchor=desk_0, members=[desk_0, office_chair_0, floor_lamp_0]) as work_nook:
    # Chair in front of desk; face toward desk. Lamp beside desk with backboard alignment.
    solver.in_front_of(source=office_chair_0, target=desk_0, clearance=chair_front_gap, alignment=Var("center"))
    solver.facing(source=office_chair_0, target=desk_0, mode="radial", mutual=False)

    solver.left_of(source=floor_lamp_0, target=desk_0, clearance=desk_lamp_gap, alignment=Var("backboard"))
    solver.align(source=floor_lamp_0, target=desk_0, angle=desk_member_align)

    # Local scaffold (anchor at origin, facing +Y)
    # Desk (1.3 x 0.7), rz=0 → NO SWAP
    desk_0.footprint = Footprint(rz=0, lx=1.3, ly=0.7)
    desk_0.pose = Pose(x=0.0, y=0.0, rz=0.0)
    desk_0.bounds = Bounds(x_min=-0.65, x_max=0.65, y_min=-0.35, y_max=0.35)

    # Chair (0.7 x 0.9), in front with 0.3m gap
    office_chair_0.footprint = Footprint(rz=0, lx=0.7, ly=0.9)
    office_chair_0.pose = Pose(x=0.0, y=1.1, rz=0.0)
    office_chair_0.bounds = Bounds(x_min=-0.35, x_max=0.35, y_min=0.65, y_max=1.55)

    # Lamp (0.3 x 0.3) to left with backboard alignment
    floor_lamp_0.footprint = Footprint(rz=0, lx=0.3, ly=0.3)
    floor_lamp_0.pose = Pose(x=-1.0, y=-0.2, rz=0.0)
    floor_lamp_0.bounds = Bounds(x_min=-1.15, x_max=-0.85, y_min=-0.35, y_max=-0.05)

# Cluster AABB (local)
work_nook.aabb = AABB(lx=1.8, ly=1.9)

# -----------------------
# Cluster: Reading Nook (armchair + side table + floor lamp)
# -----------------------
with solver.cluster(cluster_id="reading_nook", anchor=armchair_0, members=[armchair_0, side_table_0, floor_lamp_1]) as reading_nook:
    # Side table to right, lamp to left, both with backboard alignment; align orientations to armchair
    solver.right_of(source=side_table_0, target=armchair_0, clearance=reading_side_gap, alignment=Var("backboard"))
    solver.align(source=side_table_0, target=armchair_0, angle=reading_member_align)

    solver.left_of(source=floor_lamp_1, target=armchair_0, clearance=reading_side_gap, alignment=Var("backboard"))
    solver.align(source=floor_lamp_1, target=armchair_0, angle=reading_member_align)

    # Local scaffold (anchor at origin, facing +Y)
    # Armchair (0.8 x 1.0)
    armchair_0.footprint = Footprint(rz=0, lx=0.8, ly=1.0)
    armchair_0.pose = Pose(x=0.0, y=0.0, rz=0.0)
    armchair_0.bounds = Bounds(x_min=-0.4, x_max=0.4, y_min=-0.5, y_max=0.5)

    # Side table (0.5 x 0.5) to right
    side_table_0.footprint = Footprint(rz=0, lx=0.5, ly=0.5)
    side_table_0.pose = Pose(x=0.85, y=-0.25, rz=0.0)
    side_table_0.bounds = Bounds(x_min=0.6, x_max=1.1, y_min=-0.5, y_max=0.0)

    # Floor lamp (0.3 x 0.3) to left
    floor_lamp_1.footprint = Footprint(rz=0, lx=0.3, ly=0.3)
    floor_lamp_1.pose = Pose(x=-0.75, y=-0.35, rz=0.0)
    floor_lamp_1.bounds = Bounds(x_min=-0.9, x_max=-0.6, y_min=-0.5, y_max=-0.2)

# Cluster AABB (local)
reading_nook.aabb = AABB(lx=2.0, ly=1.0)

# -----------------------
# Global placement (BLACK BOXES + independent assets)
# -----------------------

# Sleeping area: bed with nightstands against LEFT wall, centered vertically
solver.against_wall(source=sleeping_area, wall="L")
solver.vertical(source=sleeping_area, y=bed_center_y)

# Work nook: place in the BR corner, backed to bottom wall (faces +Y)
solver.corner(source=work_nook, corner="BR", wall="B")

# Reading nook: place in the TR corner, backed to top wall (faces -Y)
solver.corner(source=reading_nook, corner="TR", wall="T")

# Dresser (sideboard): centered on wall opposite bed (RIGHT wall), faces into room
solver.against_wall(source=sideboard_0, wall="R")
solver.vertical(source=sideboard_0, y=dresser_center_y)

# Wardrobe: along the RIGHT wall too, separated vertically from the dresser
solver.against_wall(source=wardrobe_0, wall="R")
solver.vertical(source=wardrobe_0, y=wardrobe_center_y)

# Full-length mirror: on the TOP wall (remaining wall space), near left side
solver.against_wall(source=mirror_0, wall="T")
solver.horizontal(source=mirror_0, x=mirror_top_x)

# Potted plants fill unused corners
solver.corner(source=potted_plant_0, corner="TL", wall="T")
solver.corner(source=potted_plant_1, corner="BL", wall="B")

# -----------------------
# Global scaffold (footprints/poses/bounds)
# -----------------------

# Sleeping area: local AABB (3.7, 2.6), against L → rz=-90 → SWAP → footprint (2.6, 3.7)
sleeping_area.footprint = Footprint(rz=-90, lx=2.6, ly=3.7)
sleeping_area.pose = Pose(x=1.3, y=2.5, rz=-90)
sleeping_area.bounds = Bounds(x_min=0.0, x_max=2.6, y_min=0.65, y_max=4.35)

# Work nook: corner BR, wall B → rz=0 → NO SWAP → (1.8, 1.9)
work_nook.footprint = Footprint(rz=0, lx=1.8, ly=1.9)
work_nook.pose = Pose(x=4.1, y=0.95, rz=0.0)
work_nook.bounds = Bounds(x_min=3.2, x_max=5.0, y_min=0.0, y_max=1.9)

# Reading nook: corner TR, wall T → rz=180 → NO SWAP → (2.0, 1.0)
reading_nook.footprint = Footprint(rz=180, lx=2.0, ly=1.0)
reading_nook.pose = Pose(x=4.0, y=4.5, rz=180.0)
reading_nook.bounds = Bounds(x_min=3.0, x_max=5.0, y_min=4.0, y_max=5.0)

# Sideboard on RIGHT wall: rz=90 → SWAP → footprint (0.5, 2.4)
sideboard_0.footprint = Footprint(rz=90, lx=0.5, ly=2.4)
sideboard_0.pose = Pose(x=4.75, y=2.5, rz=90.0)
sideboard_0.bounds = Bounds(x_min=4.5, x_max=5.0, y_min=1.3, y_max=3.7)

# Wardrobe on RIGHT wall: rz=90 → SWAP → footprint (0.7, 2.1)
wardrobe_0.footprint = Footprint(rz=90, lx=0.7, ly=2.1)
wardrobe_0.pose = Pose(x=4.65, y=1.6, rz=90.0)
wardrobe_0.bounds = Bounds(x_min=4.3, x_max=5.0, y_min=0.55, y_max=2.65)

# Mirror on TOP wall: rz=180 → NO SWAP → footprint (0.7, 0.4)
mirror_0.footprint = Footprint(rz=180, lx=0.7, ly=0.4)
mirror_0.pose = Pose(x=1.0, y=4.8, rz=180.0)
mirror_0.bounds = Bounds(x_min=0.65, x_max=1.35, y_min=4.6, y_max=5.0)

# Potted plant TL (against T): rz=180 → NO SWAP → (0.2, 0.3)
potted_plant_0.footprint = Footprint(rz=180, lx=0.2, ly=0.3)
potted_plant_0.pose = Pose(x=0.1, y=4.85, rz=180.0)
potted_plant_0.bounds = Bounds(x_min=0.0, x_max=0.2, y_min=4.7, y_max=5.0)

# Potted plant BL (against B): rz=0 → NO SWAP → (0.2, 0.3)
potted_plant_1.footprint = Footprint(rz=0, lx=0.2, ly=0.3)
potted_plant_1.pose = Pose(x=0.1, y=0.15, rz=0.0)
potted_plant_1.bounds = Bounds(x_min=0.0, x_max=0.2, y_min=0.0, y_max=0.3)

# Final checks (reasoning):
# - Each cluster/member and independent asset has both placement and orientation constraints.
# - Sleeping area occupies left wall; dresser centered on right wall; work nook at BR; reading nook at TR; wardrobe on right wall (offset from dresser); mirror on top wall; plants in TL and BL corners.
# - Circulation: Center of room remains open for movement.
