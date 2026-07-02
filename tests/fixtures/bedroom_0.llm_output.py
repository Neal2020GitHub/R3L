# Define room
room = Room(length=6.0, width=5.0)

# Derived 2D wall segments (for reference)
walls = {
    "L": ((0.0, 0.0), (0.0, room.width)),
    "R": ((room.length, 0.0), (room.length, room.width)),
    "B": ((0.0, 0.0), (room.length, 0.0)),
    "T": ((0.0, room.width), (room.length, room.width))
}

# Var declarations

# Global placement along walls
sleeping_x = Var(1.75)   # center x for sleeping cluster along top wall -> x span [0.0, 3.5]
lounge_x = Var(2.6)      # center x for sofa cluster along bottom wall -> x span [1.0, 4.2]

# Bed-wardrobe local arrangement
ns_clearance = Var(0.2)
ns_alignment = Var("backboard")
ns_angle = Var(0.0)

# Workstation local arrangement
chair_clearance = Var(0.3)
chair_alignment = Var("center")

# Coffee table in front of sofa (GLOBAL)
ct_clearance = Var(0.2)
ct_alignment = Var("center")
ct_align_angle = Var(0.0)

# Armchair placement and orientation (GLOBAL)
ac_front_clear = Var(0.6)
ac_align = Var("center")
armchair_x = Var(5.2)  # push the armchair to the right side, clear of the workstation bbox



# Cluster: Sleeping Area (bed + wardrobe)
with solver.cluster(cluster_id="sleeping_area", anchor=bed_0, members=[bed_0, wardrobe_0]) as sleeping_area:
    # Wardrobe beside bed; backs flush (good for top-wall mounting later)
    solver.right_of(source=wardrobe_0, target=bed_0, clearance=ns_clearance, alignment=ns_alignment)
    solver.align(source=wardrobe_0, target=bed_0, angle=ns_angle)

    # Local scaffold (anchor at origin, facing +Y)

    # Bed (anchor): size=(2.1, 2.1), rz=0 → footprint (2.1, 2.1)
    bed_0.footprint = Footprint(rz=0, lx=2.1, ly=2.1)
    bed_0.pose = Pose(x=0.0, y=0.0, rz=0.0)
    bed_0.bounds = Bounds(x_min=-1.05, x_max=1.05, y_min=-1.05, y_max=1.05)

    # Wardrobe: size=(1.2, 0.5), rz=0 → footprint (1.2, 0.5)
    # right_of with 0.2m: bed.right=1.05 → wardrobe.left=1.25 → center.x=1.85
    # backboard alignment: y_min equal to bed.y_min=-1.05 → center.y=-0.80
    wardrobe_0.footprint = Footprint(rz=0, lx=1.2, ly=0.5)
    wardrobe_0.pose = Pose(x=1.85, y=-0.80, rz=0.0)
    wardrobe_0.bounds = Bounds(x_min=1.25, x_max=2.45, y_min=-1.05, y_max=-0.55)

# Cluster AABB (local): spans x [-1.05, 2.45] → lx=3.5; y [-1.05, 1.05] → ly=2.1
sleeping_area.aabb = AABB(lx=3.5, ly=2.1)



# Cluster: Lounge Area (sofa only; table and armchair placed globally for flexibility)
with solver.cluster(cluster_id="lounge_area", anchor=sectional_sofa_0, members=[sectional_sofa_0]) as lounge_area:
    # Local scaffold: sofa anchor at origin, facing +Y
    # Sectional sofa: size=(3.2, 1.7), rz=0 → footprint (3.2, 1.7)
    sectional_sofa_0.footprint = Footprint(rz=0, lx=3.2, ly=1.7)
    sectional_sofa_0.pose = Pose(x=0.0, y=0.0, rz=0.0)
    sectional_sofa_0.bounds = Bounds(x_min=-1.6, x_max=1.6, y_min=-0.85, y_max=0.85)

# Cluster AABB (local): sofa only → lx=3.2, ly=1.7
lounge_area.aabb = AABB(lx=3.2, ly=1.7)



# Cluster: Workstation (desk + gaming chair)
with solver.cluster(cluster_id="workstation", anchor=office_desk_0, members=[office_desk_0, gaming_chair_0]) as workstation:
    # Chair in front of desk, facing the desk
    solver.in_front_of(source=gaming_chair_0, target=office_desk_0, clearance=chair_clearance, alignment=chair_alignment)
    solver.facing(source=gaming_chair_0, target=office_desk_0, mode="radial", mutual=False)

    # Local scaffold
    # Desk (anchor): size=(1.5, 0.8), rz=0 → footprint (1.5, 0.8)
    office_desk_0.footprint = Footprint(rz=0, lx=1.5, ly=0.8)
    office_desk_0.pose = Pose(x=0.0, y=0.0, rz=0.0)
    office_desk_0.bounds = Bounds(x_min=-0.75, x_max=0.75, y_min=-0.4, y_max=0.4)

    # Gaming chair: size=(0.7, 0.7), rz≈0/180 (square, orientation doesn't affect footprint)
    # in_front_of with 0.3m: desk.front y_max=0.4 → chair.y_min=0.7 → center.y=1.05
    gaming_chair_0.footprint = Footprint(rz=0, lx=0.7, ly=0.7)
    gaming_chair_0.pose = Pose(x=0.0, y=1.05, rz=0.0)
    gaming_chair_0.bounds = Bounds(x_min=-0.35, x_max=0.35, y_min=0.7, y_max=1.4)

# Cluster AABB (local): x [-0.75, 0.75] → lx=1.5; y [-0.4, 1.4] → ly=1.8
workstation.aabb = AABB(lx=1.5, ly=1.8)



# Global constraints: place cluster handles and independent assets

# Sleeping area along the back (top) wall
solver.against_wall(source=sleeping_area, wall="T")
solver.horizontal(source=sleeping_area, x=sleeping_x)

# Sofa along the bottom wall; table/chair placed relative to it globally
solver.against_wall(source=lounge_area, wall="B")
solver.horizontal(source=lounge_area, x=lounge_x)

# Workstation in the front-right corner, backed to the right wall to face toward center (-X)
solver.corner(source=workstation, corner="BR", wall="R")

# Coffee table in front of sofa (sofa faces +Y), keep it centered
solver.in_front_of(source=coffee_table_0, target=lounge_area, clearance=ct_clearance, alignment=ct_alignment)
solver.align(source=coffee_table_0, target=lounge_area, angle=ct_align_angle)

# Armchair in front of sofa but shifted right via absolute x; angle it toward the coffee table
solver.in_front_of(source=armchair_0, target=lounge_area, clearance=ac_front_clear, alignment=ac_align)
solver.horizontal(source=armchair_0, x=armchair_x)
solver.facing(source=armchair_0, target=coffee_table_0, mode="radial", mutual=False)



# Global scaffold (room frame)

# Sleeping area: local AABB (3.5, 2.1), against T → rz=180 (no swap)
sleeping_area.footprint = Footprint(rz=180, lx=3.5, ly=2.1)
sleeping_area.pose = Pose(x=1.75, y=5.0 - 2.1/2, rz=180)  # y=3.95
sleeping_area.bounds = Bounds(x_min=1.75-1.75, x_max=1.75+1.75, y_min=3.95-1.05, y_max=3.95+1.05)  # [0.0,3.5] x [2.9,5.0]

# Lounge area: local AABB (3.2, 1.7), against B → rz=0 (no swap)
lounge_area.footprint = Footprint(rz=0, lx=3.2, ly=1.7)
lounge_area.pose = Pose(x=2.6, y=1.7/2, rz=0)  # y=0.85
lounge_area.bounds = Bounds(x_min=2.6-1.6, x_max=2.6+1.6, y_min=0.0, y_max=1.7)  # [1.0,4.2] x [0.0,1.7]

# Workstation: local AABB (1.5, 1.8), corner BR with wall R → rz=90 → SWAP → (1.8, 1.5)
workstation.footprint = Footprint(rz=90, lx=1.8, ly=1.5)
workstation.pose = Pose(x=6.0 - 1.8/2, y=1.5/2, rz=90)  # x=5.1, y=0.75
workstation.bounds = Bounds(x_min=5.1-0.9, x_max=5.1+0.9, y_min=0.0, y_max=1.5)  # [4.2,6.0] x [0.0,1.5]

# Coffee table (independent): size=(1.0,1.0), rz=0 → footprint (1.0,1.0)
# Placed in front of sofa: y_min = lounge.y_max(=1.7) + 0.2 = 1.9 → center.y = 2.4; x aligned to lounge center (2.6)
coffee_table_0.footprint = Footprint(rz=0, lx=1.0, ly=1.0)
coffee_table_0.pose = Pose(x=2.6, y=2.4, rz=0.0)
coffee_table_0.bounds = Bounds(x_min=2.6-0.5, x_max=2.6+0.5, y_min=1.9, y_max=2.9)  # [2.1,3.1] x [1.9,2.9]

# Armchair (independent): size=(1.4,1.2). Face the coffee table (approx rz≈101° from +Y).
# Rotate to rz=101° → footprint using general formula:
#  lx ≈ |1.4*cos(101)| + |1.2*sin(101)| ≈ 1.445
#  ly ≈ |1.4*sin(101)| + |1.2*cos(101)| ≈ 1.603
# In front of sofa: y_min = lounge.y_max(=1.7) + 0.6 = 2.3 → center.y ≈ 2.3 + 1.603/2 ≈ 3.1016
# Horizontal pin x=5.2 to sit right of the seating group.
armchair_0.footprint = Footprint(rz=101.0, lx=1.445, ly=1.603)
armchair_0.pose = Pose(x=5.2, y=3.1016, rz=101.0)
armchair_0.bounds = Bounds(x_min=5.2-1.445/2, x_max=5.2+1.445/2, y_min=3.1016-1.603/2, y_max=3.1016+1.603/2)  # [4.4775,5.9225] x [2.3000,3.9032]

# Final validations (conceptual checks):
# - No global bounds overlaps among sleeping_area, lounge_area, workstation, coffee_table_0, armchair_0.
# - Required orientations satisfied: wall/corner for clusters; coffee table aligned to sofa; armchair faces coffee table.
# - Circulation: clear walkway between bottom seating band (≤1.7m) and bed band (≥2.9m); desk occupies BR corner.
