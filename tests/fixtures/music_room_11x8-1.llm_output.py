# Define room
room = Room(length=11.0, width=8.0)

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

# Piano placement (bottom-center), drum station (top-center)
center_x = Var(5.5)
piano_y = Var(1.6)
drum_x = Var(5.5)

# Shared station layout vars (LOCAL)
ms_clear = Var(0.35)          # chair -> music stand front gap
side_inst_clear = Var(0.2)    # chair -> side instrument gap
stand_align = Var("center")
side_align = Var("center")
stand_face_angle = Var(0.0)   # kept for potential offsets (unused in constraints)
inst_align_angle = Var(0.0)

# Drum station local clearances (LOCAL)
ds_chair_clear = Var(0.5)
ds_stand_side = Var(0.25)

# Audience sofas (bottom wall)
sofa_left_x = Var(2.2)
sofa_right_x = Var(8.8)

# Brass storage (right wall) Ys
mi0_y = Var(7.2)
mi1_y = Var(6.2)
tuba2_y = Var(5.2)
tuba3_y = Var(4.2)

# Spare chairs (left wall) Ys
fc19_y = Var(1.0)
fc20_y = Var(1.6)
fc21_y = Var(2.2)
fc22_y = Var(2.8)
fc23_y = Var(3.4)

# Spare stands (left wall) Ys
ms19_y = Var(4.2)
ms20_y = Var(4.9)
ms21_y = Var(5.6)
ms22_y = Var(6.3)
ms23_y = Var(7.0)

# Orientation helper: piano faces opposite to drum set (drum faces -Y, piano +Y)
piano_face_up_angle = Var(180.0)

# -----------------------
# Station cluster coordinate seeds (GLOBAL positions, room frame)
# We place 18 station handles with explicit x,y centers and orient each to face the piano (radial).
# Row 1 (nearest to piano), Row 2 (middle), Row 3 (back)
# Left wing: two double-bass stations; Center: two violin stations; Right wing: two cello stations.
# -----------------------

# Row 1 (r≈2.7m) angles approx: L: [-65°, -35°], C: [-10°, +10°], R: [+35°, +65°]
r1_bass_L1_x = Var(3.053); r1_bass_L1_y = Var(2.741)  # θ=-65°
r1_bass_L2_x = Var(3.950); r1_bass_L2_y = Var(3.812)  # θ=-35°
r1_vln_C1_x  = Var(5.031); r1_vln_C1_y  = Var(4.259)  # θ=-10°
r1_vln_C2_x  = Var(5.969); r1_vln_C2_y  = Var(4.259)  # θ=+10°
r1_cello_R1_x= Var(7.050); r1_cello_R1_y= Var(3.812)  # θ=+35°
r1_cello_R2_x= Var(7.947); r1_cello_R2_y= Var(2.741)  # θ=+65°

# Row 2 (r≈3.7m) angles approx: L: [-65°, -40°], C: [-25°, +25°], R: [+40°, +65°]
r2_bass_L1_x = Var(2.148); r2_bass_L1_y = Var(3.164)  # θ=-65°
r2_bass_L2_x = Var(3.121); r2_bass_L2_y = Var(4.434)  # θ=-40°
r2_vln_C1_x  = Var(3.936); r2_vln_C1_y  = Var(4.953)  # θ=-25°
r2_vln_C2_x  = Var(7.064); r2_vln_C2_y  = Var(4.953)  # θ=+25°
r2_cello_R1_x= Var(7.879); r2_cello_R1_y= Var(4.434)  # θ=+40°
r2_cello_R2_x= Var(8.852); r2_cello_R2_y= Var(3.164)  # θ=+65°

# Row 3 (r≈4.7m) angles approx: L: [-65°, -45°], C: [-35°, +35°], R: [+45°, +65°]
r3_bass_L1_x = Var(1.240); r3_bass_L1_y = Var(3.585)  # θ=-65°
r3_bass_L2_x = Var(2.177); r3_bass_L2_y = Var(4.923)  # θ=-45°
r3_vln_C1_x  = Var(2.802); r3_vln_C1_y  = Var(5.451)  # θ=-35°
r3_vln_C2_x  = Var(8.198); r3_vln_C2_y  = Var(5.451)  # θ=+35°
r3_cello_R1_x= Var(8.823); r3_cello_R1_y= Var(4.923)  # θ=+45°
r3_cello_R2_x= Var(9.760); r3_cello_R2_y= Var(3.585)  # θ=+65°

# -----------------------
# LOCAL CLUSTERS: station assembly (chair anchor, music stand in front; side instrument for bass/cello)
# -----------------------

# Row 1
with solver.cluster(cluster_id="r1_bass_L1", anchor=folding_chair_0, members=[folding_chair_0, music_stand_0, double_bass_0]) as r1_bass_L1:
    solver.in_front_of(source=music_stand_0, target=folding_chair_0, clearance=ms_clear, alignment=Var("center"))
    solver.facing(source=music_stand_0, target=folding_chair_0, mode="radial", mutual=False)
    solver.left_of(source=double_bass_0, target=folding_chair_0, clearance=side_inst_clear, alignment=side_align)
    solver.align(source=double_bass_0, target=folding_chair_0, angle=inst_align_angle)

    # Local scaffold (anchor at origin, rz=0)
    folding_chair_0.footprint = Footprint(rz=0, lx=0.5, ly=0.4)
    folding_chair_0.pose = Pose(x=0.0, y=0.0, rz=0.0)
    folding_chair_0.bounds = Bounds(x_min=-0.25, x_max=0.25, y_min=-0.2, y_max=0.2)

    music_stand_0.footprint = Footprint(rz=180, lx=0.8, ly=0.4)
    music_stand_0.pose = Pose(x=0.0, y=0.75, rz=180.0)
    music_stand_0.bounds = Bounds(x_min=-0.4, x_max=0.4, y_min=0.55, y_max=0.95)

    double_bass_0.footprint = Footprint(rz=0, lx=0.7, ly=0.5)
    double_bass_0.pose = Pose(x=-0.8, y=0.0, rz=0.0)
    double_bass_0.bounds = Bounds(x_min=-1.15, x_max=-0.45, y_min=-0.25, y_max=0.25)

# Cluster AABB
r1_bass_L1.aabb = AABB(lx=1.55, ly=1.2)

with solver.cluster(cluster_id="r1_bass_L2", anchor=folding_chair_1, members=[folding_chair_1, music_stand_1, double_bass_1]) as r1_bass_L2:
    solver.in_front_of(source=music_stand_1, target=folding_chair_1, clearance=ms_clear, alignment=Var("center"))
    solver.facing(source=music_stand_1, target=folding_chair_1, mode="radial", mutual=False)
    solver.left_of(source=double_bass_1, target=folding_chair_1, clearance=side_inst_clear, alignment=side_align)
    solver.align(source=double_bass_1, target=folding_chair_1, angle=inst_align_angle)

    folding_chair_1.footprint = Footprint(rz=0, lx=0.5, ly=0.4)
    folding_chair_1.pose = Pose(x=0.0, y=0.0, rz=0.0)
    folding_chair_1.bounds = Bounds(x_min=-0.25, x_max=0.25, y_min=-0.2, y_max=0.2)

    music_stand_1.footprint = Footprint(rz=180, lx=0.8, ly=0.4)
    music_stand_1.pose = Pose(x=0.0, y=0.75, rz=180.0)
    music_stand_1.bounds = Bounds(x_min=-0.4, x_max=0.4, y_min=0.55, y_max=0.95)

    double_bass_1.footprint = Footprint(rz=0, lx=0.7, ly=0.5)
    double_bass_1.pose = Pose(x=-0.8, y=0.0, rz=0.0)
    double_bass_1.bounds = Bounds(x_min=-1.15, x_max=-0.45, y_min=-0.25, y_max=0.25)

r1_bass_L2.aabb = AABB(lx=1.55, ly=1.2)

with solver.cluster(cluster_id="r1_vln_C1", anchor=folding_chair_2, members=[folding_chair_2, music_stand_2]) as r1_vln_C1:
    solver.in_front_of(source=music_stand_2, target=folding_chair_2, clearance=ms_clear, alignment=stand_align)
    solver.facing(source=music_stand_2, target=folding_chair_2, mode="radial", mutual=False)

    folding_chair_2.footprint = Footprint(rz=0, lx=0.5, ly=0.4)
    folding_chair_2.pose = Pose(x=0.0, y=0.0, rz=0.0)
    folding_chair_2.bounds = Bounds(x_min=-0.25, x_max=0.25, y_min=-0.2, y_max=0.2)

    music_stand_2.footprint = Footprint(rz=180, lx=0.8, ly=0.4)
    music_stand_2.pose = Pose(x=0.0, y=0.75, rz=180.0)
    music_stand_2.bounds = Bounds(x_min=-0.4, x_max=0.4, y_min=0.55, y_max=0.95)

r1_vln_C1.aabb = AABB(lx=0.8, ly=1.15)

with solver.cluster(cluster_id="r1_vln_C2", anchor=folding_chair_3, members=[folding_chair_3, music_stand_3]) as r1_vln_C2:
    solver.in_front_of(source=music_stand_3, target=folding_chair_3, clearance=ms_clear, alignment=stand_align)
    solver.facing(source=music_stand_3, target=folding_chair_3, mode="radial", mutual=False)

    folding_chair_3.footprint = Footprint(rz=0, lx=0.5, ly=0.4)
    folding_chair_3.pose = Pose(x=0.0, y=0.0, rz=0.0)
    folding_chair_3.bounds = Bounds(x_min=-0.25, x_max=0.25, y_min=-0.2, y_max=0.2)

    music_stand_3.footprint = Footprint(rz=180, lx=0.8, ly=0.4)
    music_stand_3.pose = Pose(x=0.0, y=0.75, rz=180.0)
    music_stand_3.bounds = Bounds(x_min=-0.4, x_max=0.4, y_min=0.55, y_max=0.95)

r1_vln_C2.aabb = AABB(lx=0.8, ly=1.15)

with solver.cluster(cluster_id="r1_cello_R1", anchor=folding_chair_4, members=[folding_chair_4, music_stand_4, cello_0]) as r1_cello_R1:
    solver.in_front_of(source=music_stand_4, target=folding_chair_4, clearance=ms_clear, alignment=stand_align)
    solver.facing(source=music_stand_4, target=folding_chair_4, mode="radial", mutual=False)
    solver.right_of(source=cello_0, target=folding_chair_4, clearance=side_inst_clear, alignment=side_align)
    solver.align(source=cello_0, target=folding_chair_4, angle=inst_align_angle)

    folding_chair_4.footprint = Footprint(rz=0, lx=0.5, ly=0.4)
    folding_chair_4.pose = Pose(x=0.0, y=0.0, rz=0.0)
    folding_chair_4.bounds = Bounds(x_min=-0.25, x_max=0.25, y_min=-0.2, y_max=0.2)

    music_stand_4.footprint = Footprint(rz=180, lx=0.8, ly=0.4)
    music_stand_4.pose = Pose(x=0.0, y=0.75, rz=180.0)
    music_stand_4.bounds = Bounds(x_min=-0.4, x_max=0.4, y_min=0.55, y_max=0.95)

    cello_0.footprint = Footprint(rz=0, lx=0.4, ly=0.8)
    cello_0.pose = Pose(x=0.8, y=0.0, rz=0.0)
    cello_0.bounds = Bounds(x_min=0.6, x_max=1.0, y_min=-0.4, y_max=0.4)

r1_cello_R1.aabb = AABB(lx=1.4, ly=1.35)

with solver.cluster(cluster_id="r1_cello_R2", anchor=folding_chair_5, members=[folding_chair_5, music_stand_5, cello_1]) as r1_cello_R2:
    solver.in_front_of(source=music_stand_5, target=folding_chair_5, clearance=ms_clear, alignment=stand_align)
    solver.facing(source=music_stand_5, target=folding_chair_5, mode="radial", mutual=False)
    solver.right_of(source=cello_1, target=folding_chair_5, clearance=side_inst_clear, alignment=side_align)
    solver.align(source=cello_1, target=folding_chair_5, angle=inst_align_angle)

    folding_chair_5.footprint = Footprint(rz=0, lx=0.5, ly=0.4)
    folding_chair_5.pose = Pose(x=0.0, y=0.0, rz=0.0)
    folding_chair_5.bounds = Bounds(x_min=-0.25, x_max=0.25, y_min=-0.2, y_max=0.2)

    music_stand_5.footprint = Footprint(rz=180, lx=0.8, ly=0.4)
    music_stand_5.pose = Pose(x=0.0, y=0.75, rz=180.0)
    music_stand_5.bounds = Bounds(x_min=-0.4, x_max=0.4, y_min=0.55, y_max=0.95)

    cello_1.footprint = Footprint(rz=0, lx=0.4, ly=0.8)
    cello_1.pose = Pose(x=0.8, y=0.0, rz=0.0)
    cello_1.bounds = Bounds(x_min=0.6, x_max=1.0, y_min=-0.4, y_max=0.4)

r1_cello_R2.aabb = AABB(lx=1.4, ly=1.35)

# Row 2
with solver.cluster(cluster_id="r2_bass_L1", anchor=folding_chair_6, members=[folding_chair_6, music_stand_6, double_bass_2]) as r2_bass_L1:
    solver.in_front_of(source=music_stand_6, target=folding_chair_6, clearance=ms_clear, alignment=stand_align)
    solver.facing(source=music_stand_6, target=folding_chair_6, mode="radial", mutual=False)
    solver.left_of(source=double_bass_2, target=folding_chair_6, clearance=side_inst_clear, alignment=side_align)
    solver.align(source=double_bass_2, target=folding_chair_6, angle=inst_align_angle)

    folding_chair_6.footprint = Footprint(rz=0, lx=0.5, ly=0.4)
    folding_chair_6.pose = Pose(x=0.0, y=0.0, rz=0.0)
    folding_chair_6.bounds = Bounds(x_min=-0.25, x_max=0.25, y_min=-0.2, y_max=0.2)

    music_stand_6.footprint = Footprint(rz=180, lx=0.8, ly=0.4)
    music_stand_6.pose = Pose(x=0.0, y=0.75, rz=180.0)
    music_stand_6.bounds = Bounds(x_min=-0.4, x_max=0.4, y_min=0.55, y_max=0.95)

    double_bass_2.footprint = Footprint(rz=0, lx=0.7, ly=0.5)
    double_bass_2.pose = Pose(x=-0.8, y=0.0, rz=0.0)
    double_bass_2.bounds = Bounds(x_min=-1.15, x_max=-0.45, y_min=-0.25, y_max=0.25)

r2_bass_L1.aabb = AABB(lx=1.55, ly=1.2)

with solver.cluster(cluster_id="r2_bass_L2", anchor=folding_chair_7, members=[folding_chair_7, music_stand_7, double_bass_3]) as r2_bass_L2:
    solver.in_front_of(source=music_stand_7, target=folding_chair_7, clearance=ms_clear, alignment=stand_align)
    solver.facing(source=music_stand_7, target=folding_chair_7, mode="radial", mutual=False)
    solver.left_of(source=double_bass_3, target=folding_chair_7, clearance=side_inst_clear, alignment=side_align)
    solver.align(source=double_bass_3, target=folding_chair_7, angle=inst_align_angle)

    folding_chair_7.footprint = Footprint(rz=0, lx=0.5, ly=0.4)
    folding_chair_7.pose = Pose(x=0.0, y=0.0, rz=0.0)
    folding_chair_7.bounds = Bounds(x_min=-0.25, x_max=0.25, y_min=-0.2, y_max=0.2)

    music_stand_7.footprint = Footprint(rz=180, lx=0.8, ly=0.4)
    music_stand_7.pose = Pose(x=0.0, y=0.75, rz=180.0)
    music_stand_7.bounds = Bounds(x_min=-0.4, x_max=0.4, y_min=0.55, y_max=0.95)

    double_bass_3.footprint = Footprint(rz=0, lx=0.7, ly=0.5)
    double_bass_3.pose = Pose(x=-0.8, y=0.0, rz=0.0)
    double_bass_3.bounds = Bounds(x_min=-1.15, x_max=-0.45, y_min=-0.25, y_max=0.25)

r2_bass_L2.aabb = AABB(lx=1.55, ly=1.2)

with solver.cluster(cluster_id="r2_vln_C1", anchor=folding_chair_8, members=[folding_chair_8, music_stand_8]) as r2_vln_C1:
    solver.in_front_of(source=music_stand_8, target=folding_chair_8, clearance=ms_clear, alignment=stand_align)
    solver.facing(source=music_stand_8, target=folding_chair_8, mode="radial", mutual=False)

    folding_chair_8.footprint = Footprint(rz=0, lx=0.5, ly=0.4)
    folding_chair_8.pose = Pose(x=0.0, y=0.0, rz=0.0)
    folding_chair_8.bounds = Bounds(x_min=-0.25, x_max=0.25, y_min=-0.2, y_max=0.2)

    music_stand_8.footprint = Footprint(rz=180, lx=0.8, ly=0.4)
    music_stand_8.pose = Pose(x=0.0, y=0.75, rz=180.0)
    music_stand_8.bounds = Bounds(x_min=-0.4, x_max=0.4, y_min=0.55, y_max=0.95)

r2_vln_C1.aabb = AABB(lx=0.8, ly=1.15)

with solver.cluster(cluster_id="r2_vln_C2", anchor=folding_chair_9, members=[folding_chair_9, music_stand_9]) as r2_vln_C2:
    solver.in_front_of(source=music_stand_9, target=folding_chair_9, clearance=ms_clear, alignment=stand_align)
    solver.facing(source=music_stand_9, target=folding_chair_9, mode="radial", mutual=False)

    folding_chair_9.footprint = Footprint(rz=0, lx=0.5, ly=0.4)
    folding_chair_9.pose = Pose(x=0.0, y=0.0, rz=0.0)
    folding_chair_9.bounds = Bounds(x_min=-0.25, x_max=0.25, y_min=-0.2, y_max=0.2)

    music_stand_9.footprint = Footprint(rz=180, lx=0.8, ly=0.4)
    music_stand_9.pose = Pose(x=0.0, y=0.75, rz=180.0)
    music_stand_9.bounds = Bounds(x_min=-0.4, x_max=0.4, y_min=0.55, y_max=0.95)

r2_vln_C2.aabb = AABB(lx=0.8, ly=1.15)

with solver.cluster(cluster_id="r2_cello_R1", anchor=folding_chair_10, members=[folding_chair_10, music_stand_10, cello_2]) as r2_cello_R1:
    solver.in_front_of(source=music_stand_10, target=folding_chair_10, clearance=ms_clear, alignment=stand_align)
    solver.facing(source=music_stand_10, target=folding_chair_10, mode="radial", mutual=False)
    solver.right_of(source=cello_2, target=folding_chair_10, clearance=side_inst_clear, alignment=side_align)
    solver.align(source=cello_2, target=folding_chair_10, angle=inst_align_angle)

    folding_chair_10.footprint = Footprint(rz=0, lx=0.5, ly=0.4)
    folding_chair_10.pose = Pose(x=0.0, y=0.0, rz=0.0)
    folding_chair_10.bounds = Bounds(x_min=-0.25, x_max=0.25, y_min=-0.2, y_max=0.2)

    music_stand_10.footprint = Footprint(rz=180, lx=0.8, ly=0.4)
    music_stand_10.pose = Pose(x=0.0, y=0.75, rz=180.0)
    music_stand_10.bounds = Bounds(x_min=-0.4, x_max=0.4, y_min=0.55, y_max=0.95)

    cello_2.footprint = Footprint(rz=0, lx=0.4, ly=0.8)
    cello_2.pose = Pose(x=0.8, y=0.0, rz=0.0)
    cello_2.bounds = Bounds(x_min=0.6, x_max=1.0, y_min=-0.4, y_max=0.4)

r2_cello_R1.aabb = AABB(lx=1.4, ly=1.35)

with solver.cluster(cluster_id="r2_cello_R2", anchor=folding_chair_11, members=[folding_chair_11, music_stand_11, cello_3]) as r2_cello_R2:
    solver.in_front_of(source=music_stand_11, target=folding_chair_11, clearance=ms_clear, alignment=stand_align)
    solver.facing(source=music_stand_11, target=folding_chair_11, mode="radial", mutual=False)
    solver.right_of(source=cello_3, target=folding_chair_11, clearance=side_inst_clear, alignment=side_align)
    solver.align(source=cello_3, target=folding_chair_11, angle=inst_align_angle)

    folding_chair_11.footprint = Footprint(rz=0, lx=0.5, ly=0.4)
    folding_chair_11.pose = Pose(x=0.0, y=0.0, rz=0.0)
    folding_chair_11.bounds = Bounds(x_min=-0.25, x_max=0.25, y_min=-0.2, y_max=0.2)

    music_stand_11.footprint = Footprint(rz=180, lx=0.8, ly=0.4)
    music_stand_11.pose = Pose(x=0.0, y=0.75, rz=180.0)
    music_stand_11.bounds = Bounds(x_min=-0.4, x_max=0.4, y_min=0.55, y_max=0.95)

    cello_3.footprint = Footprint(rz=0, lx=0.4, ly=0.8)
    cello_3.pose = Pose(x=0.8, y=0.0, rz=0.0)
    cello_3.bounds = Bounds(x_min=0.6, x_max=1.0, y_min=-0.4, y_max=0.4)

r2_cello_R2.aabb = AABB(lx=1.4, ly=1.35)

# Row 3
with solver.cluster(cluster_id="r3_bass_L1", anchor=folding_chair_12, members=[folding_chair_12, music_stand_12, double_bass_4]) as r3_bass_L1:
    solver.in_front_of(source=music_stand_12, target=folding_chair_12, clearance=ms_clear, alignment=stand_align)
    solver.facing(source=music_stand_12, target=folding_chair_12, mode="radial", mutual=False)
    solver.left_of(source=double_bass_4, target=folding_chair_12, clearance=side_inst_clear, alignment=side_align)
    solver.align(source=double_bass_4, target=folding_chair_12, angle=inst_align_angle)

    folding_chair_12.footprint = Footprint(rz=0, lx=0.5, ly=0.4)
    folding_chair_12.pose = Pose(x=0.0, y=0.0, rz=0.0)
    folding_chair_12.bounds = Bounds(x_min=-0.25, x_max=0.25, y_min=-0.2, y_max=0.2)

    music_stand_12.footprint = Footprint(rz=180, lx=0.8, ly=0.4)
    music_stand_12.pose = Pose(x=0.0, y=0.75, rz=180.0)
    music_stand_12.bounds = Bounds(x_min=-0.4, x_max=0.4, y_min=0.55, y_max=0.95)

    double_bass_4.footprint = Footprint(rz=0, lx=0.7, ly=0.5)
    double_bass_4.pose = Pose(x=-0.8, y=0.0, rz=0.0)
    double_bass_4.bounds = Bounds(x_min=-1.15, x_max=-0.45, y_min=-0.25, y_max=0.25)

r3_bass_L1.aabb = AABB(lx=1.55, ly=1.2)

with solver.cluster(cluster_id="r3_bass_L2", anchor=folding_chair_13, members=[folding_chair_13, music_stand_13, double_bass_5]) as r3_bass_L2:
    solver.in_front_of(source=music_stand_13, target=folding_chair_13, clearance=ms_clear, alignment=stand_align)
    solver.facing(source=music_stand_13, target=folding_chair_13, mode="radial", mutual=False)
    solver.left_of(source=double_bass_5, target=folding_chair_13, clearance=side_inst_clear, alignment=side_align)
    solver.align(source=double_bass_5, target=folding_chair_13, angle=inst_align_angle)

    folding_chair_13.footprint = Footprint(rz=0, lx=0.5, ly=0.4)
    folding_chair_13.pose = Pose(x=0.0, y=0.0, rz=0.0)
    folding_chair_13.bounds = Bounds(x_min=-0.25, x_max=0.25, y_min=-0.2, y_max=0.2)

    music_stand_13.footprint = Footprint(rz=180, lx=0.8, ly=0.4)
    music_stand_13.pose = Pose(x=0.0, y=0.75, rz=180.0)
    music_stand_13.bounds = Bounds(x_min=-0.4, x_max=0.4, y_min=0.55, y_max=0.95)

    double_bass_5.footprint = Footprint(rz=0, lx=0.7, ly=0.5)
    double_bass_5.pose = Pose(x=-0.8, y=0.0, rz=0.0)
    double_bass_5.bounds = Bounds(x_min=-1.15, x_max=-0.45, y_min=-0.25, y_max=0.25)

r3_bass_L2.aabb = AABB(lx=1.55, ly=1.2)

with solver.cluster(cluster_id="r3_vln_C1", anchor=folding_chair_14, members=[folding_chair_14, music_stand_14]) as r3_vln_C1:
    solver.in_front_of(source=music_stand_14, target=folding_chair_14, clearance=ms_clear, alignment=stand_align)
    solver.facing(source=music_stand_14, target=folding_chair_14, mode="radial", mutual=False)

    folding_chair_14.footprint = Footprint(rz=0, lx=0.5, ly=0.4)
    folding_chair_14.pose = Pose(x=0.0, y=0.0, rz=0.0)
    folding_chair_14.bounds = Bounds(x_min=-0.25, x_max=0.25, y_min=-0.2, y_max=0.2)

    music_stand_14.footprint = Footprint(rz=180, lx=0.8, ly=0.4)
    music_stand_14.pose = Pose(x=0.0, y=0.75, rz=180.0)
    music_stand_14.bounds = Bounds(x_min=-0.4, x_max=0.4, y_min=0.55, y_max=0.95)

r3_vln_C1.aabb = AABB(lx=0.8, ly=1.15)

with solver.cluster(cluster_id="r3_vln_C2", anchor=folding_chair_15, members=[folding_chair_15, music_stand_15]) as r3_vln_C2:
    solver.in_front_of(source=music_stand_15, target=folding_chair_15, clearance=ms_clear, alignment=stand_align)
    solver.facing(source=music_stand_15, target=folding_chair_15, mode="radial", mutual=False)

    folding_chair_15.footprint = Footprint(rz=0, lx=0.5, ly=0.4)
    folding_chair_15.pose = Pose(x=0.0, y=0.0, rz=0.0)
    folding_chair_15.bounds = Bounds(x_min=-0.25, x_max=0.25, y_min=-0.2, y_max=0.2)

    music_stand_15.footprint = Footprint(rz=180, lx=0.8, ly=0.4)
    music_stand_15.pose = Pose(x=0.0, y=0.75, rz=180.0)
    music_stand_15.bounds = Bounds(x_min=-0.4, x_max=0.4, y_min=0.55, y_max=0.95)

r3_vln_C2.aabb = AABB(lx=0.8, ly=1.15)

with solver.cluster(cluster_id="r3_cello_R1", anchor=folding_chair_16, members=[folding_chair_16, music_stand_16, cello_4]) as r3_cello_R1:
    solver.in_front_of(source=music_stand_16, target=folding_chair_16, clearance=ms_clear, alignment=stand_align)
    solver.facing(source=music_stand_16, target=folding_chair_16, mode="radial", mutual=False)
    solver.right_of(source=cello_4, target=folding_chair_16, clearance=side_inst_clear, alignment=side_align)
    solver.align(source=cello_4, target=folding_chair_16, angle=inst_align_angle)

    folding_chair_16.footprint = Footprint(rz=0, lx=0.5, ly=0.4)
    folding_chair_16.pose = Pose(x=0.0, y=0.0, rz=0.0)
    folding_chair_16.bounds = Bounds(x_min=-0.25, x_max=0.25, y_min=-0.2, y_max=0.2)

    music_stand_16.footprint = Footprint(rz=180, lx=0.8, ly=0.4)
    music_stand_16.pose = Pose(x=0.0, y=0.75, rz=180.0)
    music_stand_16.bounds = Bounds(x_min=-0.4, x_max=0.4, y_min=0.55, y_max=0.95)

    cello_4.footprint = Footprint(rz=0, lx=0.4, ly=0.8)
    cello_4.pose = Pose(x=0.8, y=0.0, rz=0.0)
    cello_4.bounds = Bounds(x_min=0.6, x_max=1.0, y_min=-0.4, y_max=0.4)

r3_cello_R1.aabb = AABB(lx=1.4, ly=1.35)

with solver.cluster(cluster_id="r3_cello_R2", anchor=folding_chair_17, members=[folding_chair_17, music_stand_17, cello_5]) as r3_cello_R2:
    solver.in_front_of(source=music_stand_17, target=folding_chair_17, clearance=ms_clear, alignment=stand_align)
    solver.facing(source=music_stand_17, target=folding_chair_17, mode="radial", mutual=False)
    solver.right_of(source=cello_5, target=folding_chair_17, clearance=side_inst_clear, alignment=side_align)
    solver.align(source=cello_5, target=folding_chair_17, angle=inst_align_angle)

    folding_chair_17.footprint = Footprint(rz=0, lx=0.5, ly=0.4)
    folding_chair_17.pose = Pose(x=0.0, y=0.0, rz=0.0)
    folding_chair_17.bounds = Bounds(x_min=-0.25, x_max=0.25, y_min=-0.2, y_max=0.2)

    music_stand_17.footprint = Footprint(rz=180, lx=0.8, ly=0.4)
    music_stand_17.pose = Pose(x=0.0, y=0.75, rz=180.0)
    music_stand_17.bounds = Bounds(x_min=-0.4, x_max=0.4, y_min=0.55, y_max=0.95)

    cello_5.footprint = Footprint(rz=0, lx=0.4, ly=0.8)
    cello_5.pose = Pose(x=0.8, y=0.0, rz=0.0)
    cello_5.bounds = Bounds(x_min=0.6, x_max=1.0, y_min=-0.4, y_max=0.4)

r3_cello_R2.aabb = AABB(lx=1.4, ly=1.35)

# -----------------------
# Drum station as a wall-bound cluster (anchor = drum set)
# -----------------------
with solver.cluster(cluster_id="drum_station", anchor=drum_set_0, members=[drum_set_0, folding_chair_18, music_stand_18]) as drum_station:
    # Place drummer's chair behind the kit in LOCAL frame so when cluster faces -Y at top wall, chair ends up in front of the kit.
    solver.behind_of(source=folding_chair_18, target=drum_set_0, clearance=ds_chair_clear, alignment=Var("center"))
    solver.facing(source=folding_chair_18, target=drum_set_0, mode="ortho", mutual=False)
    solver.right_of(source=music_stand_18, target=folding_chair_18, clearance=ds_stand_side, alignment=Var("backboard"))
    solver.facing(source=music_stand_18, target=folding_chair_18, mode="radial", mutual=False)

    # Local scaffold
    drum_set_0.footprint = Footprint(rz=0, lx=2.4, ly=1.7)
    drum_set_0.pose = Pose(x=0.0, y=0.0, rz=0.0)
    drum_set_0.bounds = Bounds(x_min=-1.2, x_max=1.2, y_min=-0.85, y_max=0.85)

    folding_chair_18.footprint = Footprint(rz=0, lx=0.5, ly=0.4)
    # Chair center y = back edge of kit (-0.85) - clear (0.5) - chair half (0.2) = -1.55
    folding_chair_18.pose = Pose(x=0.0, y=-1.55, rz=0.0)
    folding_chair_18.bounds = Bounds(x_min=-0.25, x_max=0.25, y_min=-1.75, y_max=-1.35)

    music_stand_18.footprint = Footprint(rz=180, lx=0.8, ly=0.4)
    music_stand_18.pose = Pose(x=0.9, y=-1.55, rz=180.0)
    music_stand_18.bounds = Bounds(x_min=0.5, x_max=1.3, y_min=-1.75, y_max=-1.35)

# AABB (local) for drum station
drum_station.aabb = AABB(lx=2.5, ly=2.6)

# -----------------------
# GLOBAL INTEGRATION
# -----------------------

# Grand piano prominent bottom-center, facing +Y toward orchestra
solver.horizontal(source=grand_piano_0, x=center_x)
solver.vertical(source=grand_piano_0, y=piano_y)
# orientation: align opposite to drum station's facing (drum faces -Y when backed to top wall)
solver.align(source=grand_piano_0, target=drum_station, angle=piano_face_up_angle)

# Drummer at top-center against the wall
solver.against_wall(source=drum_station, wall="T")
solver.horizontal(source=drum_station, x=drum_x)

# Harp in top-left corner
solver.corner(source=harp_0, corner="TL", wall="L")

# Two sofas at bottom for audience
solver.against_wall(source=sofa_0, wall="B")
solver.horizontal(source=sofa_0, x=sofa_left_x)

solver.against_wall(source=sofa_1, wall="B")
solver.horizontal(source=sofa_1, x=sofa_right_x)

# Brass storage along right wall
solver.against_wall(source=musical_instrument_0, wall="R")
solver.vertical(source=musical_instrument_0, y=mi0_y)

solver.against_wall(source=musical_instrument_1, wall="R")
solver.vertical(source=musical_instrument_1, y=mi1_y)

solver.against_wall(source=musical_instrument_2, wall="R")
solver.vertical(source=musical_instrument_2, y=tuba2_y)

solver.against_wall(source=musical_instrument_3, wall="R")
solver.vertical(source=musical_instrument_3, y=tuba3_y)

# Spare chairs along left wall
solver.against_wall(source=folding_chair_19, wall="L")
solver.vertical(source=folding_chair_19, y=fc19_y)

solver.against_wall(source=folding_chair_20, wall="L")
solver.vertical(source=folding_chair_20, y=fc20_y)

solver.against_wall(source=folding_chair_21, wall="L")
solver.vertical(source=folding_chair_21, y=fc21_y)

solver.against_wall(source=folding_chair_22, wall="L")
solver.vertical(source=folding_chair_22, y=fc22_y)

solver.against_wall(source=folding_chair_23, wall="L")
solver.vertical(source=folding_chair_23, y=fc23_y)

# Spare stands along left wall (higher band)
solver.against_wall(source=music_stand_19, wall="L")
solver.vertical(source=music_stand_19, y=ms19_y)

solver.against_wall(source=music_stand_20, wall="L")
solver.vertical(source=music_stand_20, y=ms20_y)

solver.against_wall(source=music_stand_21, wall="L")
solver.vertical(source=music_stand_21, y=ms21_y)

solver.against_wall(source=music_stand_22, wall="L")
solver.vertical(source=music_stand_22, y=ms22_y)

solver.against_wall(source=music_stand_23, wall="L")
solver.vertical(source=music_stand_23, y=ms23_y)

# -----------------------
# Global placement of station handles: positions + radial facing toward the piano
# -----------------------

# Row 1
solver.horizontal(source=r1_bass_L1, x=r1_bass_L1_x); solver.vertical(source=r1_bass_L1, y=r1_bass_L1_y)
solver.facing(source=r1_bass_L1, target=grand_piano_0, mode="radial", mutual=False)

solver.horizontal(source=r1_bass_L2, x=r1_bass_L2_x); solver.vertical(source=r1_bass_L2, y=r1_bass_L2_y)
solver.facing(source=r1_bass_L2, target=grand_piano_0, mode="radial", mutual=False)

solver.horizontal(source=r1_vln_C1, x=r1_vln_C1_x); solver.vertical(source=r1_vln_C1, y=r1_vln_C1_y)
solver.facing(source=r1_vln_C1, target=grand_piano_0, mode="radial", mutual=False)

solver.horizontal(source=r1_vln_C2, x=r1_vln_C2_x); solver.vertical(source=r1_vln_C2, y=r1_vln_C2_y)
solver.facing(source=r1_vln_C2, target=grand_piano_0, mode="radial", mutual=False)

solver.horizontal(source=r1_cello_R1, x=r1_cello_R1_x); solver.vertical(source=r1_cello_R1, y=r1_cello_R1_y)
solver.facing(source=r1_cello_R1, target=grand_piano_0, mode="radial", mutual=False)

solver.horizontal(source=r1_cello_R2, x=r1_cello_R2_x); solver.vertical(source=r1_cello_R2, y=r1_cello_R2_y)
solver.facing(source=r1_cello_R2, target=grand_piano_0, mode="radial", mutual=False)

# Row 2
solver.horizontal(source=r2_bass_L1, x=r2_bass_L1_x); solver.vertical(source=r2_bass_L1, y=r2_bass_L1_y)
solver.facing(source=r2_bass_L1, target=grand_piano_0, mode="radial", mutual=False)

solver.horizontal(source=r2_bass_L2, x=r2_bass_L2_x); solver.vertical(source=r2_bass_L2, y=r2_bass_L2_y)
solver.facing(source=r2_bass_L2, target=grand_piano_0, mode="radial", mutual=False)

solver.horizontal(source=r2_vln_C1, x=r2_vln_C1_x); solver.vertical(source=r2_vln_C1, y=r2_vln_C1_y)
solver.facing(source=r2_vln_C1, target=grand_piano_0, mode="radial", mutual=False)

solver.horizontal(source=r2_vln_C2, x=r2_vln_C2_x); solver.vertical(source=r2_vln_C2, y=r2_vln_C2_y)
solver.facing(source=r2_vln_C2, target=grand_piano_0, mode="radial", mutual=False)

solver.horizontal(source=r2_cello_R1, x=r2_cello_R1_x); solver.vertical(source=r2_cello_R1, y=r2_cello_R1_y)
solver.facing(source=r2_cello_R1, target=grand_piano_0, mode="radial", mutual=False)

solver.horizontal(source=r2_cello_R2, x=r2_cello_R2_x); solver.vertical(source=r2_cello_R2, y=r2_cello_R2_y)
solver.facing(source=r2_cello_R2, target=grand_piano_0, mode="radial", mutual=False)

# Row 3
solver.horizontal(source=r3_bass_L1, x=r3_bass_L1_x); solver.vertical(source=r3_bass_L1, y=r3_bass_L1_y)
solver.facing(source=r3_bass_L1, target=grand_piano_0, mode="radial", mutual=False)

solver.horizontal(source=r3_bass_L2, x=r3_bass_L2_x); solver.vertical(source=r3_bass_L2, y=r3_bass_L2_y)
solver.facing(source=r3_bass_L2, target=grand_piano_0, mode="radial", mutual=False)

solver.horizontal(source=r3_vln_C1, x=r3_vln_C1_x); solver.vertical(source=r3_vln_C1, y=r3_vln_C1_y)
solver.facing(source=r3_vln_C1, target=grand_piano_0, mode="radial", mutual=False)

solver.horizontal(source=r3_vln_C2, x=r3_vln_C2_x); solver.vertical(source=r3_vln_C2, y=r3_vln_C2_y)
solver.facing(source=r3_vln_C2, target=grand_piano_0, mode="radial", mutual=False)

solver.horizontal(source=r3_cello_R1, x=r3_cello_R1_x); solver.vertical(source=r3_cello_R1, y=r3_cello_R1_y)
solver.facing(source=r3_cello_R1, target=grand_piano_0, mode="radial", mutual=False)

solver.horizontal(source=r3_cello_R2, x=r3_cello_R2_x); solver.vertical(source=r3_cello_R2, y=r3_cello_R2_y)
solver.facing(source=r3_cello_R2, target=grand_piano_0, mode="radial", mutual=False)

# -----------------------
# Global scaffold (key exemplars to verify map; optimizer will refine)
# -----------------------

# Grand piano scaffold: size=(1.6, 2.4), rz≈0 -> NO SWAP
grand_piano_0.footprint = Footprint(rz=0, lx=1.6, ly=2.4)
grand_piano_0.pose = Pose(x=5.5, y=1.6, rz=0.0)
grand_piano_0.bounds = Bounds(x_min=4.7, x_max=6.3, y_min=0.4, y_max=2.8)

# Drum station scaffold: AABB (2.5, 2.6), rz=180 -> NO SWAP
drum_station.footprint = Footprint(rz=180, lx=2.5, ly=2.6)
drum_station.pose = Pose(x=5.5, y=6.7, rz=180.0)
drum_station.bounds = Bounds(x_min=4.25, x_max=6.75, y_min=5.4, y_max=8.0)

# Harp (TL corner, back to L -> facing +X => rz=-90)
harp_0.footprint = Footprint(rz=-90, lx=0.3, ly=1.2)
harp_0.pose = Pose(x=0.15, y=7.4, rz=-90.0)
harp_0.bounds = Bounds(x_min=0.0, x_max=0.3, y_min=6.8, y_max=8.0)

# Sofas (against B, face +Y rz=0)
sofa_0.footprint = Footprint(rz=0, lx=2.3, ly=0.8)
sofa_0.pose = Pose(x=2.2, y=0.4, rz=0.0)
sofa_0.bounds = Bounds(x_min=1.05, x_max=3.35, y_min=0.0, y_max=0.8)

sofa_1.footprint = Footprint(rz=0, lx=2.3, ly=0.8)
sofa_1.pose = Pose(x=8.8, y=0.4, rz=0.0)
sofa_1.bounds = Bounds(x_min=7.65, x_max=9.95, y_min=0.0, y_max=0.8)

# Brass storage along right wall (rz=90 -> swap)
musical_instrument_0.footprint = Footprint(rz=90, lx=0.5, ly=0.3)
musical_instrument_0.pose = Pose(x=10.75, y=7.2, rz=90.0)
musical_instrument_0.bounds = Bounds(x_min=10.5, x_max=11.0, y_min=7.05, y_max=7.35)

musical_instrument_1.footprint = Footprint(rz=90, lx=0.5, ly=0.3)
musical_instrument_1.pose = Pose(x=10.75, y=6.2, rz=90.0)
musical_instrument_1.bounds = Bounds(x_min=10.5, x_max=11.0, y_min=6.05, y_max=6.35)

musical_instrument_2.footprint = Footprint(rz=90, lx=0.4, ly=0.6)
musical_instrument_2.pose = Pose(x=10.8, y=5.2, rz=90.0)
musical_instrument_2.bounds = Bounds(x_min=10.6, x_max=11.0, y_min=4.9, y_max=5.5)

musical_instrument_3.footprint = Footprint(rz=90, lx=0.4, ly=0.6)
musical_instrument_3.pose = Pose(x=10.8, y=4.2, rz=90.0)
musical_instrument_3.bounds = Bounds(x_min=10.6, x_max=11.0, y_min=3.9, y_max=4.5)

# Spare chairs along left wall (rz=-90)
folding_chair_19.footprint = Footprint(rz=-90, lx=0.4, ly=0.5)
folding_chair_19.pose = Pose(x=0.2, y=1.0, rz=-90.0)
folding_chair_19.bounds = Bounds(x_min=0.0, x_max=0.4, y_min=0.75, y_max=1.25)

folding_chair_20.footprint = Footprint(rz=-90, lx=0.4, ly=0.5)
folding_chair_20.pose = Pose(x=0.2, y=1.6, rz=-90.0)
folding_chair_20.bounds = Bounds(x_min=0.0, x_max=0.4, y_min=1.35, y_max=1.85)

folding_chair_21.footprint = Footprint(rz=-90, lx=0.4, ly=0.5)
folding_chair_21.pose = Pose(x=0.2, y=2.2, rz=-90.0)
folding_chair_21.bounds = Bounds(x_min=0.0, x_max=0.4, y_min=1.95, y_max=2.45)

folding_chair_22.footprint = Footprint(rz=-90, lx=0.4, ly=0.5)
folding_chair_22.pose = Pose(x=0.2, y=2.8, rz=-90.0)
folding_chair_22.bounds = Bounds(x_min=0.0, x_max=0.4, y_min=2.55, y_max=3.05)

folding_chair_23.footprint = Footprint(rz=-90, lx=0.4, ly=0.5)
folding_chair_23.pose = Pose(x=0.2, y=3.4, rz=-90.0)
folding_chair_23.bounds = Bounds(x_min=0.0, x_max=0.4, y_min=3.15, y_max=3.65)

# Spare stands along left wall (rz=-90 -> swap)
music_stand_19.footprint = Footprint(rz=-90, lx=0.4, ly=0.8)
music_stand_19.pose = Pose(x=0.2, y=4.2, rz=-90.0)
music_stand_19.bounds = Bounds(x_min=0.0, x_max=0.4, y_min=3.8, y_max=4.6)

music_stand_20.footprint = Footprint(rz=-90, lx=0.4, ly=0.8)
music_stand_20.pose = Pose(x=0.2, y=4.9, rz=-90.0)
music_stand_20.bounds = Bounds(x_min=0.0, x_max=0.4, y_min=4.5, y_max=5.3)

music_stand_21.footprint = Footprint(rz=-90, lx=0.4, ly=0.8)
music_stand_21.pose = Pose(x=0.2, y=5.6, rz=-90.0)
music_stand_21.bounds = Bounds(x_min=0.0, x_max=0.4, y_min=5.2, y_max=6.0)

music_stand_22.footprint = Footprint(rz=-90, lx=0.4, ly=0.8)
music_stand_22.pose = Pose(x=0.2, y=6.3, rz=-90.0)
music_stand_22.bounds = Bounds(x_min=0.0, x_max=0.4, y_min=5.9, y_max=6.7)

music_stand_23.footprint = Footprint(rz=-90, lx=0.4, ly=0.8)
music_stand_23.pose = Pose(x=0.2, y=7.0, rz=-90.0)
music_stand_23.bounds = Bounds(x_min=0.0, x_max=0.4, y_min=6.6, y_max=7.4)

# -----------------------
# Global scaffold for all 18 station cluster handles (AABBs, rz=0 for scaffold; constraints will rotate radially)
# -----------------------

# Row 1
r1_bass_L1.footprint = Footprint(rz=0, lx=1.55, ly=1.2)
r1_bass_L1.pose = Pose(x=3.053, y=2.741, rz=0.0)
r1_bass_L1.bounds = Bounds(x_min=2.278, x_max=3.828, y_min=2.141, y_max=3.341)

r1_bass_L2.footprint = Footprint(rz=0, lx=1.55, ly=1.2)
r1_bass_L2.pose = Pose(x=3.95, y=3.812, rz=0.0)
r1_bass_L2.bounds = Bounds(x_min=3.175, x_max=4.725, y_min=3.212, y_max=4.412)

r1_vln_C1.footprint = Footprint(rz=0, lx=0.8, ly=1.15)
r1_vln_C1.pose = Pose(x=5.031, y=4.259, rz=0.0)
r1_vln_C1.bounds = Bounds(x_min=4.631, x_max=5.431, y_min=3.684, y_max=4.834)

r1_vln_C2.footprint = Footprint(rz=0, lx=0.8, ly=1.15)
r1_vln_C2.pose = Pose(x=5.969, y=4.259, rz=0.0)
r1_vln_C2.bounds = Bounds(x_min=5.569, x_max=6.369, y_min=3.684, y_max=4.834)

r1_cello_R1.footprint = Footprint(rz=0, lx=1.4, ly=1.35)
r1_cello_R1.pose = Pose(x=7.05, y=3.812, rz=0.0)
r1_cello_R1.bounds = Bounds(x_min=6.35, x_max=7.75, y_min=3.137, y_max=4.487)

r1_cello_R2.footprint = Footprint(rz=0, lx=1.4, ly=1.35)
r1_cello_R2.pose = Pose(x=7.947, y=2.741, rz=0.0)
r1_cello_R2.bounds = Bounds(x_min=7.247, x_max=8.647, y_min=2.066, y_max=3.416)

# Row 2
r2_bass_L1.footprint = Footprint(rz=0, lx=1.55, ly=1.2)
r2_bass_L1.pose = Pose(x=2.148, y=3.164, rz=0.0)
r2_bass_L1.bounds = Bounds(x_min=1.373, x_max=2.923, y_min=2.564, y_max=3.764)

r2_bass_L2.footprint = Footprint(rz=0, lx=1.55, ly=1.2)
r2_bass_L2.pose = Pose(x=3.121, y=4.434, rz=0.0)
r2_bass_L2.bounds = Bounds(x_min=2.346, x_max=3.896, y_min=3.834, y_max=5.034)

r2_vln_C1.footprint = Footprint(rz=0, lx=0.8, ly=1.15)
r2_vln_C1.pose = Pose(x=3.936, y=4.953, rz=0.0)
r2_vln_C1.bounds = Bounds(x_min=3.536, x_max=4.336, y_min=4.378, y_max=5.528)

r2_vln_C2.footprint = Footprint(rz=0, lx=0.8, ly=1.15)
r2_vln_C2.pose = Pose(x=7.064, y=4.953, rz=0.0)
r2_vln_C2.bounds = Bounds(x_min=6.664, x_max=7.464, y_min=4.378, y_max=5.528)

r2_cello_R1.footprint = Footprint(rz=0, lx=1.4, ly=1.35)
r2_cello_R1.pose = Pose(x=7.879, y=4.434, rz=0.0)
r2_cello_R1.bounds = Bounds(x_min=7.179, x_max=8.579, y_min=3.759, y_max=5.109)

r2_cello_R2.footprint = Footprint(rz=0, lx=1.4, ly=1.35)
r2_cello_R2.pose = Pose(x=8.852, y=3.164, rz=0.0)
r2_cello_R2.bounds = Bounds(x_min=8.152, x_max=9.552, y_min=2.489, y_max=3.839)

# Row 3
r3_bass_L1.footprint = Footprint(rz=0, lx=1.55, ly=1.2)
r3_bass_L1.pose = Pose(x=1.24, y=3.585, rz=0.0)
r3_bass_L1.bounds = Bounds(x_min=0.465, x_max=2.015, y_min=2.985, y_max=4.185)

r3_bass_L2.footprint = Footprint(rz=0, lx=1.55, ly=1.2)
r3_bass_L2.pose = Pose(x=2.177, y=4.923, rz=0.0)
r3_bass_L2.bounds = Bounds(x_min=1.402, x_max=2.952, y_min=4.323, y_max=5.523)

r3_vln_C1.footprint = Footprint(rz=0, lx=0.8, ly=1.15)
r3_vln_C1.pose = Pose(x=2.802, y=5.451, rz=0.0)
r3_vln_C1.bounds = Bounds(x_min=2.402, x_max=3.202, y_min=4.876, y_max=6.026)

r3_vln_C2.footprint = Footprint(rz=0, lx=0.8, ly=1.15)
r3_vln_C2.pose = Pose(x=8.198, y=5.451, rz=0.0)
r3_vln_C2.bounds = Bounds(x_min=7.798, x_max=8.998, y_min=4.876, y_max=6.026)

r3_cello_R1.footprint = Footprint(rz=0, lx=1.4, ly=1.35)
r3_cello_R1.pose = Pose(x=8.823, y=4.923, rz=0.0)
r3_cello_R1.bounds = Bounds(x_min=8.123, x_max=9.523, y_min=4.248, y_max=5.598)

r3_cello_R2.footprint = Footprint(rz=0, lx=1.4, ly=1.35)
r3_cello_R2.pose = Pose(x=9.76, y=3.585, rz=0.0)
r3_cello_R2.bounds = Bounds(x_min=9.06, x_max=10.46, y_min=2.91, y_max=4.26)
