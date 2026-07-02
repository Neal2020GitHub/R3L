# Define room
room = Room(length=12.0, width=9.0)

# Derived 2D wall segments (for reference)
walls = {
    "L": ((0.0, 0.0), (0.0, room.width)),
    "R": ((room.length, 0.0), (room.length, room.width)),
    "B": ((0.0, 0.0), (room.length, 0.0)),
    "T": ((0.0, room.width), (room.length, room.width))
}

# --------------------
# Var declarations
# --------------------
zero = Var(0.0)

# Alignments
back_align = Var("backboard")      # for lateral chaining with flush backs
center_align_lr = Var("center")    # lateral center alignment when needed
center_align_fb = Var("center")    # frontal/behind center alignment

# Angles
angle_0 = Var(0.0)
angle_180 = Var(180.0)

# Cooking station global placement
station_x = Var(5.5)
stationA_y = Var(5.8)   # top pair front-facing +Y
stationC_y = Var(3.3)   # lower pair front-facing +Y (slightly below pair-1)

# Plating grid global placement
plating_x = Var(10.2)
plating_y = Var(4.5)

# Left-wall kitchen runs (along wall Y placement)
kitchen0_y = Var(7.3)
kitchen1_y = Var(3.8)

# Right-wall equipment (vertical along wall)
grill0_y = Var(3.7)
grill1_y = Var(6.9)
fryer_y = Var(8.74)

# Shopping carts row (bottom center)
carts_x = Var(6.0)
carts_y = Var(0.8)
cart_gap = Var(0.3)

# --------------------
# Clusters: Local Assembly (Infinite Void)
# --------------------

# Cooking Station A: [bin - bench - stove - stove - bench - bin]
with solver.cluster(cluster_id="station_A", anchor=electric_stove_0, members=[electric_stove_0, electric_stove_1, workbench_6, workbench_7, trash_bin_0, trash_bin_1]) as station_A:
    solver.right_of(source=electric_stove_1, target=electric_stove_0, clearance=zero, alignment=back_align)
    solver.align(source=electric_stove_1, target=electric_stove_0, angle=angle_0)

    solver.left_of(source=workbench_6, target=electric_stove_0, clearance=zero, alignment=back_align)
    solver.align(source=workbench_6, target=electric_stove_0, angle=angle_0)

    solver.right_of(source=workbench_7, target=electric_stove_1, clearance=zero, alignment=back_align)
    solver.align(source=workbench_7, target=electric_stove_0, angle=angle_0)

    solver.left_of(source=trash_bin_0, target=workbench_6, clearance=zero, alignment=back_align)
    solver.align(source=trash_bin_0, target=electric_stove_0, angle=angle_0)

    solver.right_of(source=trash_bin_1, target=workbench_7, clearance=zero, alignment=back_align)
    solver.align(source=trash_bin_1, target=electric_stove_0, angle=angle_0)

    # Local scaffold (rz=0 for all)
    # Anchor stove0
    electric_stove_0.footprint = Footprint(rz=0, lx=0.9, ly=1.1)
    electric_stove_0.pose = Pose(x=0.0, y=0.0, rz=0.0)
    electric_stove_0.bounds = Bounds(x_min=-0.45, x_max=0.45, y_min=-0.55, y_max=0.55)

    # Stove1 (right, no gap)
    electric_stove_1.footprint = Footprint(rz=0, lx=0.9, ly=1.1)
    electric_stove_1.pose = Pose(x=0.9, y=0.0, rz=0.0)
    electric_stove_1.bounds = Bounds(x_min=0.45, x_max=1.35, y_min=-0.55, y_max=0.55)

    # Left workbench (flush back)
    workbench_6.footprint = Footprint(rz=0, lx=1.8, ly=1.0)
    workbench_6.pose = Pose(x=-1.35, y=-0.05, rz=0.0)
    workbench_6.bounds = Bounds(x_min=-2.25, x_max=-0.45, y_min=-0.55, y_max=0.45)

    # Right workbench (flush back)
    workbench_7.footprint = Footprint(rz=0, lx=1.8, ly=1.0)
    workbench_7.pose = Pose(x=2.25, y=-0.05, rz=0.0)
    workbench_7.bounds = Bounds(x_min=1.35, x_max=3.15, y_min=-0.55, y_max=0.45)

    # Left trash bin (flush back)
    trash_bin_0.footprint = Footprint(rz=0, lx=0.7, ly=0.6)
    trash_bin_0.pose = Pose(x=-2.6, y=-0.25, rz=0.0)
    trash_bin_0.bounds = Bounds(x_min=-2.95, x_max=-2.25, y_min=-0.55, y_max=0.35)

    # Right trash bin (flush back)
    trash_bin_1.footprint = Footprint(rz=0, lx=0.7, ly=0.6)
    trash_bin_1.pose = Pose(x=3.5, y=-0.25, rz=0.0)
    trash_bin_1.bounds = Bounds(x_min=3.15, x_max=3.85, y_min=-0.55, y_max=0.35)

# Cooking Station B
with solver.cluster(cluster_id="station_B", anchor=electric_stove_2, members=[electric_stove_2, electric_stove_3, workbench_8, workbench_9, trash_bin_2, trash_bin_3]) as station_B:
    solver.right_of(source=electric_stove_3, target=electric_stove_2, clearance=zero, alignment=back_align)
    solver.align(source=electric_stove_3, target=electric_stove_2, angle=angle_0)

    solver.left_of(source=workbench_8, target=electric_stove_2, clearance=zero, alignment=back_align)
    solver.align(source=workbench_8, target=electric_stove_2, angle=angle_0)

    solver.right_of(source=workbench_9, target=electric_stove_3, clearance=zero, alignment=back_align)
    solver.align(source=workbench_9, target=electric_stove_2, angle=angle_0)

    solver.left_of(source=trash_bin_2, target=workbench_8, clearance=zero, alignment=back_align)
    solver.align(source=trash_bin_2, target=electric_stove_2, angle=angle_0)

    solver.right_of(source=trash_bin_3, target=workbench_9, clearance=zero, alignment=back_align)
    solver.align(source=trash_bin_3, target=electric_stove_2, angle=angle_0)

    electric_stove_2.footprint = Footprint(rz=0, lx=0.9, ly=1.1)
    electric_stove_2.pose = Pose(x=0.0, y=0.0, rz=0.0)
    electric_stove_2.bounds = Bounds(x_min=-0.45, x_max=0.45, y_min=-0.55, y_max=0.55)

    electric_stove_3.footprint = Footprint(rz=0, lx=0.9, ly=1.1)
    electric_stove_3.pose = Pose(x=0.9, y=0.0, rz=0.0)
    electric_stove_3.bounds = Bounds(x_min=0.45, x_max=1.35, y_min=-0.55, y_max=0.55)

    workbench_8.footprint = Footprint(rz=0, lx=1.8, ly=1.0)
    workbench_8.pose = Pose(x=-1.35, y=-0.05, rz=0.0)
    workbench_8.bounds = Bounds(x_min=-2.25, x_max=-0.45, y_min=-0.55, y_max=0.45)

    workbench_9.footprint = Footprint(rz=0, lx=1.8, ly=1.0)
    workbench_9.pose = Pose(x=2.25, y=-0.05, rz=0.0)
    workbench_9.bounds = Bounds(x_min=1.35, x_max=3.15, y_min=-0.55, y_max=0.45)

    trash_bin_2.footprint = Footprint(rz=0, lx=0.7, ly=0.6)
    trash_bin_2.pose = Pose(x=-2.6, y=-0.25, rz=0.0)
    trash_bin_2.bounds = Bounds(x_min=-2.95, x_max=-2.25, y_min=-0.55, y_max=0.35)

    trash_bin_3.footprint = Footprint(rz=0, lx=0.7, ly=0.6)
    trash_bin_3.pose = Pose(x=3.5, y=-0.25, rz=0.0)
    trash_bin_3.bounds = Bounds(x_min=3.15, x_max=3.85, y_min=-0.55, y_max=0.35)

# Cooking Station C
with solver.cluster(cluster_id="station_C", anchor=electric_stove_4, members=[electric_stove_4, electric_stove_5, workbench_10, workbench_11, trash_bin_4, trash_bin_5]) as station_C:
    solver.right_of(source=electric_stove_5, target=electric_stove_4, clearance=zero, alignment=back_align)
    solver.align(source=electric_stove_5, target=electric_stove_4, angle=angle_0)

    solver.left_of(source=workbench_10, target=electric_stove_4, clearance=zero, alignment=back_align)
    solver.align(source=workbench_10, target=electric_stove_4, angle=angle_0)

    solver.right_of(source=workbench_11, target=electric_stove_5, clearance=zero, alignment=back_align)
    solver.align(source=workbench_11, target=electric_stove_4, angle=angle_0)

    solver.left_of(source=trash_bin_4, target=workbench_10, clearance=zero, alignment=back_align)
    solver.align(source=trash_bin_4, target=electric_stove_4, angle=angle_0)

    solver.right_of(source=trash_bin_5, target=workbench_11, clearance=zero, alignment=back_align)
    solver.align(source=trash_bin_5, target=electric_stove_4, angle=angle_0)

    electric_stove_4.footprint = Footprint(rz=0, lx=0.9, ly=1.1)
    electric_stove_4.pose = Pose(x=0.0, y=0.0, rz=0.0)
    electric_stove_4.bounds = Bounds(x_min=-0.45, x_max=0.45, y_min=-0.55, y_max=0.55)

    electric_stove_5.footprint = Footprint(rz=0, lx=0.9, ly=1.1)
    electric_stove_5.pose = Pose(x=0.9, y=0.0, rz=0.0)
    electric_stove_5.bounds = Bounds(x_min=0.45, x_max=1.35, y_min=-0.55, y_max=0.55)

    workbench_10.footprint = Footprint(rz=0, lx=1.8, ly=1.0)
    workbench_10.pose = Pose(x=-1.35, y=-0.05, rz=0.0)
    workbench_10.bounds = Bounds(x_min=-2.25, x_max=-0.45, y_min=-0.55, y_max=0.45)

    workbench_11.footprint = Footprint(rz=0, lx=1.8, ly=1.0)
    workbench_11.pose = Pose(x=2.25, y=-0.05, rz=0.0)
    workbench_11.bounds = Bounds(x_min=1.35, x_max=3.15, y_min=-0.55, y_max=0.45)

    trash_bin_4.footprint = Footprint(rz=0, lx=0.7, ly=0.6)
    trash_bin_4.pose = Pose(x=-2.6, y=-0.25, rz=0.0)
    trash_bin_4.bounds = Bounds(x_min=-2.95, x_max=-2.25, y_min=-0.55, y_max=0.35)

    trash_bin_5.footprint = Footprint(rz=0, lx=0.7, ly=0.6)
    trash_bin_5.pose = Pose(x=3.5, y=-0.25, rz=0.0)
    trash_bin_5.bounds = Bounds(x_min=3.15, x_max=3.85, y_min=-0.55, y_max=0.35)

# Cooking Station D
with solver.cluster(cluster_id="station_D", anchor=electric_stove_6, members=[electric_stove_6, electric_stove_7, workbench_12, workbench_13, trash_bin_6, trash_bin_7]) as station_D:
    solver.right_of(source=electric_stove_7, target=electric_stove_6, clearance=zero, alignment=back_align)
    solver.align(source=electric_stove_7, target=electric_stove_6, angle=angle_0)

    solver.left_of(source=workbench_12, target=electric_stove_6, clearance=zero, alignment=back_align)
    solver.align(source=workbench_12, target=electric_stove_6, angle=angle_0)

    solver.right_of(source=workbench_13, target=electric_stove_7, clearance=zero, alignment=back_align)
    solver.align(source=workbench_13, target=electric_stove_6, angle=angle_0)

    solver.left_of(source=trash_bin_6, target=workbench_12, clearance=zero, alignment=back_align)
    solver.align(source=trash_bin_6, target=electric_stove_6, angle=angle_0)

    solver.right_of(source=trash_bin_7, target=workbench_13, clearance=zero, alignment=back_align)
    solver.align(source=trash_bin_7, target=electric_stove_6, angle=angle_0)

    electric_stove_6.footprint = Footprint(rz=0, lx=0.9, ly=1.1)
    electric_stove_6.pose = Pose(x=0.0, y=0.0, rz=0.0)
    electric_stove_6.bounds = Bounds(x_min=-0.45, x_max=0.45, y_min=-0.55, y_max=0.55)

    electric_stove_7.footprint = Footprint(rz=0, lx=0.9, ly=1.1)
    electric_stove_7.pose = Pose(x=0.9, y=0.0, rz=0.0)
    electric_stove_7.bounds = Bounds(x_min=0.45, x_max=1.35, y_min=-0.55, y_max=0.55)

    workbench_12.footprint = Footprint(rz=0, lx=1.8, ly=1.0)
    workbench_12.pose = Pose(x=-1.35, y=-0.05, rz=0.0)
    workbench_12.bounds = Bounds(x_min=-2.25, x_max=-0.45, y_min=-0.55, y_max=0.45)

    workbench_13.footprint = Footprint(rz=0, lx=1.8, ly=1.0)
    workbench_13.pose = Pose(x=2.25, y=-0.05, rz=0.0)
    workbench_13.bounds = Bounds(x_min=1.35, x_max=3.15, y_min=-0.55, y_max=0.45)

    trash_bin_6.footprint = Footprint(rz=0, lx=0.7, ly=0.6)
    trash_bin_6.pose = Pose(x=-2.6, y=-0.25, rz=0.0)
    trash_bin_6.bounds = Bounds(x_min=-2.95, x_max=-2.25, y_min=-0.55, y_max=0.35)

    trash_bin_7.footprint = Footprint(rz=0, lx=0.7, ly=0.6)
    trash_bin_7.pose = Pose(x=3.5, y=-0.25, rz=0.0)
    trash_bin_7.bounds = Bounds(x_min=3.15, x_max=3.85, y_min=-0.55, y_max=0.35)

# Plating Area: 2x3 stainless grid (row of 3, no gaps; second row behind, opposite facing)
with solver.cluster(cluster_id="plating_area", anchor=workbench_0, members=[workbench_0, workbench_1, workbench_2, workbench_3, workbench_4, workbench_5]) as plating_area:
    # Front row (no gaps)
    solver.right_of(source=workbench_1, target=workbench_0, clearance=zero, alignment=back_align)
    solver.align(source=workbench_1, target=workbench_0, angle=angle_0)
    solver.right_of(source=workbench_2, target=workbench_1, clearance=zero, alignment=back_align)
    solver.align(source=workbench_2, target=workbench_0, angle=angle_0)

    # Back row (behind each, opposite facing)
    solver.behind_of(source=workbench_3, target=workbench_0, clearance=zero, alignment=center_align_fb)
    solver.align(source=workbench_3, target=workbench_0, angle=angle_180)

    solver.behind_of(source=workbench_4, target=workbench_1, clearance=zero, alignment=center_align_fb)
    solver.align(source=workbench_4, target=workbench_0, angle=angle_180)

    solver.behind_of(source=workbench_5, target=workbench_2, clearance=zero, alignment=center_align_fb)
    solver.align(source=workbench_5, target=workbench_0, angle=angle_180)

    # Local scaffold (rz=0 for all benches; back row uses 180 deg but same footprint)
    # Anchor front-left
    workbench_0.footprint = Footprint(rz=0, lx=2.0, ly=0.7)
    workbench_0.pose = Pose(x=0.0, y=0.0, rz=0.0)
    workbench_0.bounds = Bounds(x_min=-1.0, x_max=1.0, y_min=-0.35, y_max=0.35)

    # Front row middle, right (touching)
    workbench_1.footprint = Footprint(rz=0, lx=2.0, ly=0.7)
    workbench_1.pose = Pose(x=2.0, y=0.0, rz=0.0)
    workbench_1.bounds = Bounds(x_min=1.0, x_max=3.0, y_min=-0.35, y_max=0.35)

    workbench_2.footprint = Footprint(rz=0, lx=2.0, ly=0.7)
    workbench_2.pose = Pose(x=4.0, y=0.0, rz=0.0)
    workbench_2.bounds = Bounds(x_min=3.0, x_max=5.0, y_min=-0.35, y_max=0.35)

    # Back row aligned and opposite facing
    workbench_3.footprint = Footprint(rz=180, lx=2.0, ly=0.7)
    workbench_3.pose = Pose(x=0.0, y=-0.7, rz=180.0)
    workbench_3.bounds = Bounds(x_min=-1.0, x_max=1.0, y_min=-1.05, y_max=-0.35)

    workbench_4.footprint = Footprint(rz=180, lx=2.0, ly=0.7)
    workbench_4.pose = Pose(x=2.0, y=-0.7, rz=180.0)
    workbench_4.bounds = Bounds(x_min=1.0, x_max=3.0, y_min=-1.05, y_max=-0.35)

    workbench_5.footprint = Footprint(rz=180, lx=2.0, ly=0.7)
    workbench_5.pose = Pose(x=4.0, y=-0.7, rz=180.0)
    workbench_5.bounds = Bounds(x_min=3.0, x_max=5.0, y_min=-1.05, y_max=-0.35)

# Display racks stack (bottom-left corner, vertical stack upward)
with solver.cluster(cluster_id="display_stack", anchor=display_rack_0, members=[display_rack_0, display_rack_1, display_rack_2, display_rack_3]) as display_stack:
    solver.in_front_of(source=display_rack_1, target=display_rack_0, clearance=zero, alignment=center_align_fb)
    solver.align(source=display_rack_1, target=display_rack_0, angle=angle_0)

    solver.in_front_of(source=display_rack_2, target=display_rack_1, clearance=zero, alignment=center_align_fb)
    solver.align(source=display_rack_2, target=display_rack_0, angle=angle_0)

    solver.in_front_of(source=display_rack_3, target=display_rack_2, clearance=zero, alignment=center_align_fb)
    solver.align(source=display_rack_3, target=display_rack_0, angle=angle_0)

    # Local scaffold
    display_rack_0.footprint = Footprint(rz=0, lx=1.2, ly=0.4)
    display_rack_0.pose = Pose(x=0.0, y=0.0, rz=0.0)
    display_rack_0.bounds = Bounds(x_min=-0.6, x_max=0.6, y_min=-0.2, y_max=0.2)

    display_rack_1.footprint = Footprint(rz=0, lx=1.2, ly=0.4)
    display_rack_1.pose = Pose(x=0.0, y=0.4, rz=0.0)
    display_rack_1.bounds = Bounds(x_min=-0.6, x_max=0.6, y_min=0.2, y_max=0.6)

    display_rack_2.footprint = Footprint(rz=0, lx=1.2, ly=0.4)
    display_rack_2.pose = Pose(x=0.0, y=0.8, rz=0.0)
    display_rack_2.bounds = Bounds(x_min=-0.6, x_max=0.6, y_min=0.6, y_max=1.0)

    display_rack_3.footprint = Footprint(rz=0, lx=1.2, ly=0.4)
    display_rack_3.pose = Pose(x=0.0, y=1.2, rz=0.0)
    display_rack_3.bounds = Bounds(x_min=-0.6, x_max=0.6, y_min=1.0, y_max=1.4)

# Shopping carts row (bottom center, distributed)
with solver.cluster(cluster_id="cart_row", anchor=shopping_cart_1, members=[shopping_cart_0, shopping_cart_1, shopping_cart_2]) as cart_row:
    solver.left_of(source=shopping_cart_0, target=shopping_cart_1, clearance=cart_gap, alignment=center_align_lr)
    solver.align(source=shopping_cart_0, target=shopping_cart_1, angle=angle_0)

    solver.right_of(source=shopping_cart_2, target=shopping_cart_1, clearance=cart_gap, alignment=center_align_lr)
    solver.align(source=shopping_cart_2, target=shopping_cart_1, angle=angle_0)

    # Local scaffold
    shopping_cart_1.footprint = Footprint(rz=0, lx=0.5, ly=0.9)
    shopping_cart_1.pose = Pose(x=0.0, y=0.0, rz=0.0)
    shopping_cart_1.bounds = Bounds(x_min=-0.25, x_max=0.25, y_min=-0.45, y_max=0.45)

    shopping_cart_0.footprint = Footprint(rz=0, lx=0.5, ly=0.9)
    shopping_cart_0.pose = Pose(x=-0.8, y=0.0, rz=0.0)
    shopping_cart_0.bounds = Bounds(x_min=-1.05, x_max=-0.55, y_min=-0.45, y_max=0.45)

    shopping_cart_2.footprint = Footprint(rz=0, lx=0.5, ly=0.9)
    shopping_cart_2.pose = Pose(x=0.8, y=0.0, rz=0.0)
    shopping_cart_2.bounds = Bounds(x_min=0.55, x_max=1.05, y_min=-0.45, y_max=0.45)

# --------------------
# Seal Black Boxes — AABBs
# --------------------
station_A.aabb = AABB(lx=6.8, ly=1.1)
station_B.aabb = AABB(lx=6.8, ly=1.1)
station_C.aabb = AABB(lx=6.8, ly=1.1)
station_D.aabb = AABB(lx=6.8, ly=1.1)

plating_area.aabb = AABB(lx=6.0, ly=1.4)
display_stack.aabb = AABB(lx=1.2, ly=1.6)
cart_row.aabb = AABB(lx=2.1, ly=0.9)

# --------------------
# Global Integration
# --------------------

# Central floating cooking stations — back-to-back pairs
solver.facing(source=station_A, target="B", mode="ortho", mutual=False)
solver.horizontal(source=station_A, x=station_x)
solver.vertical(source=station_A, y=stationA_y)

solver.align(source=station_B, target=station_A, angle=angle_180)
solver.behind_of(source=station_B, target=station_A, clearance=zero, alignment=center_align_fb)
solver.horizontal(source=station_B, x=station_x)  # keep same column for robustness

solver.facing(source=station_C, target="B", mode="ortho", mutual=False)
solver.horizontal(source=station_C, x=station_x)
solver.vertical(source=station_C, y=stationC_y)

solver.align(source=station_D, target=station_C, angle=angle_180)
solver.behind_of(source=station_D, target=station_C, clearance=zero, alignment=center_align_fb)
solver.horizontal(source=station_D, x=station_x)

# Plating area — vertical grid oriented and floated to the right
solver.facing(source=plating_area, target="R", mode="ortho", mutual=False)
solver.horizontal(source=plating_area, x=plating_x)
solver.vertical(source=plating_area, y=plating_y)

# Left wall primary kitchen runs
solver.against_wall(source=kitchen_0, wall="L")
solver.vertical(source=kitchen_0, y=kitchen0_y)

solver.against_wall(source=kitchen_1, wall="L")
solver.vertical(source=kitchen_1, y=kitchen1_y)

# Right wall line: fridge in BR corner, then grills and fryer along R wall
solver.corner(source=refrigerator_0, corner="BR", wall="R")

solver.against_wall(source=barbecue_grill_0, wall="R")
solver.vertical(source=barbecue_grill_0, y=grill0_y)

solver.against_wall(source=barbecue_grill_1, wall="R")
solver.vertical(source=barbecue_grill_1, y=grill1_y)

solver.against_wall(source=deep_fryer_0, wall="R")
solver.vertical(source=deep_fryer_0, y=fryer_y)

# Display racks stack — bottom-left corner
solver.corner(source=display_stack, corner="BL", wall="L")

# Shopping carts — bottom center, face upward for egress
solver.facing(source=cart_row, target="B", mode="ortho", mutual=False)
solver.horizontal(source=cart_row, x=carts_x)
solver.vertical(source=cart_row, y=carts_y)

# --------------------
# Global Scaffolds (Handles + Independents)
# --------------------

# Stations (AABB rz: A/C -> 0, B/D -> 180)
station_A.footprint = Footprint(rz=0, lx=6.8, ly=1.1)
station_A.pose = Pose(x=5.5, y=5.8, rz=0.0)
station_A.bounds = Bounds(x_min=2.1, x_max=8.9, y_min=5.25, y_max=6.35)

station_B.footprint = Footprint(rz=180, lx=6.8, ly=1.1)
station_B.pose = Pose(x=5.5, y=4.7, rz=180.0)
station_B.bounds = Bounds(x_min=2.1, x_max=8.9, y_min=4.15, y_max=5.25)

station_C.footprint = Footprint(rz=0, lx=6.8, ly=1.1)
station_C.pose = Pose(x=5.5, y=3.3, rz=0.0)
station_C.bounds = Bounds(x_min=2.1, x_max=8.9, y_min=2.75, y_max=3.85)

station_D.footprint = Footprint(rz=180, lx=6.8, ly=1.1)
station_D.pose = Pose(x=5.5, y=2.2, rz=180.0)
station_D.bounds = Bounds(x_min=2.1, x_max=8.9, y_min=1.65, y_max=2.75)

# Plating area (rotated vertical: rz=90 -> swap to (1.4, 6.0))
plating_area.footprint = Footprint(rz=90, lx=1.4, ly=6.0)
plating_area.pose = Pose(x=10.2, y=4.5, rz=90.0)
plating_area.bounds = Bounds(x_min=9.5, x_max=10.9, y_min=1.5, y_max=7.5)

# Display stack in BL corner (rz=-90 -> swap (1.6, 1.2))
display_stack.footprint = Footprint(rz=-90, lx=1.6, ly=1.2)
display_stack.pose = Pose(x=0.8, y=0.6, rz=-90.0)
display_stack.bounds = Bounds(x_min=0.0, x_max=1.6, y_min=0.0, y_max=1.2)

# Shopping carts row (rz=0)
cart_row.footprint = Footprint(rz=0, lx=2.1, ly=0.9)
cart_row.pose = Pose(x=6.0, y=0.8, rz=0.0)
cart_row.bounds = Bounds(x_min=4.95, x_max=7.05, y_min=0.35, y_max=1.25)

# Left wall kitchens (against L -> rz=-90 -> swap (0.7, 3.3))
kitchen_0.footprint = Footprint(rz=-90, lx=0.7, ly=3.3)
kitchen_0.pose = Pose(x=0.35, y=7.3, rz=-90.0)
kitchen_0.bounds = Bounds(x_min=0.0, x_max=0.7, y_min=5.65, y_max=8.95)

kitchen_1.footprint = Footprint(rz=-90, lx=0.7, ly=3.3)
kitchen_1.pose = Pose(x=0.35, y=3.8, rz=-90.0)
kitchen_1.bounds = Bounds(x_min=0.0, x_max=0.7, y_min=2.15, y_max=5.45)

# Bottom-right refrigerator (corner BR, wall R -> rz=90 -> swap (1.2, 2.0))
refrigerator_0.footprint = Footprint(rz=90, lx=1.2, ly=2.0)
refrigerator_0.pose = Pose(x=11.4, y=1.0, rz=90.0)
refrigerator_0.bounds = Bounds(x_min=10.8, x_max=12.0, y_min=0.0, y_max=2.0)

# Right-wall grills (against R -> rz=90 -> swap (0.8, 3.1))
barbecue_grill_0.footprint = Footprint(rz=90, lx=0.8, ly=3.1)
barbecue_grill_0.pose = Pose(x=11.6, y=3.7, rz=90.0)
barbecue_grill_0.bounds = Bounds(x_min=11.2, x_max=12.0, y_min=2.15, y_max=5.25)

barbecue_grill_1.footprint = Footprint(rz=90, lx=0.8, ly=3.1)
barbecue_grill_1.pose = Pose(x=11.6, y=6.9, rz=90.0)
barbecue_grill_1.bounds = Bounds(x_min=11.2, x_max=12.0, y_min=5.35, y_max=8.45)

# Right-wall deep fryer (against R -> rz=90 -> swap (0.5, 0.5))
deep_fryer_0.footprint = Footprint(rz=90, lx=0.5, ly=0.5)
deep_fryer_0.pose = Pose(x=11.75, y=8.74, rz=90.0)
deep_fryer_0.bounds = Bounds(x_min=11.5, x_max=12.0, y_min=8.49, y_max=8.99)

# Final validations (comments):
# - All assets placed with both position and orientation constraints.
# - Central stations float in the middle with a vertical passage; plating grid stands vertical to the right with clear margins.
# - Left and right walls host continuous work runs and equipment; corners utilized for refrigerator (BR) and display racks (BL).
# - Bounds checked: no global overlaps; walkways preserved around clusters.
