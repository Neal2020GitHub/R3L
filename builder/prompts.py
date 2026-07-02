"""LLM prompts for the Scene Builder.

TASK_PROMPT is the test-case generation prompt from the R3L paper (Appendix G.1):
given a room type, room size, and difficulty level, GPT-5 returns a layout instruction
and a list of floor objects. REGENERATE_PROMPT reverse-generates an instruction from the
objects currently placed on the canvas, in the same persona and paragraph style as
TASK_PROMPT (the paper's Requirement 3).
"""

TASK_PROMPT = """You are a professional interior designer who has designed thousands of functional and aesthetically pleasing interiors.

Given a room type, room size (length m x width m), and difficulty level (easy / medium / hard), generate a practical and realistic layout instruction for the room.

Only include key floor-standing furniture or other large floor objects. Do not include small items, wall-mounted objects, ceiling objects, or tabletop decorations.

Difficulty levels:

- easy: few objects, simple compositional spatial relations
- medium: moderate number of objects, more compositional spatial relations
- hard: more objects, complex multi-step spatial relations and stronger global coordination

Requirements:
1. First provide a concise high-level description of the overall layout and design strategy.
2. Then list the key objects. For each object, provide:

- description: a short retrieval-friendly description
- size: [length, width, height] in centimeters
- quantity: integer
- variance_type: "same" or "varied"

3. Write one coherent paragraph as the instruction. It should describe the overall layout strategy, the listed objects, and the spatial arrangement among these objects using clear relative relations such as left of, right of, in front of, behind, next to, facing, aligned with, against a wall, near a corner, or centered in.

4. The layout must be feasible, functional, and consistent with the given room type, size, and difficulty level.

Output valid JSON only:
{
  "difficulty": "easy | medium | hard",
  "instruction": "...",
  "objects": {
    "object_name": {
      "description": "...",
      "size": [100, 60, 75],
      "quantity": 1,
      "variance_type": "same"
    }
  }
}"""


# Reverse-generate a layout instruction (paper-style) from the objects currently placed on the canvas.
REGENERATE_PROMPT = """You are a professional interior designer who has designed thousands of functional and aesthetically pleasing interiors.

Given a room type, room size (length m x width m), and the floor objects currently placed in the room, write a layout instruction describing how the room is organized.

Room type: ROOM_TYPE
Room size: ROOM_LENGTH m x ROOM_WIDTH m

Objects currently placed:
OBJECT_LIST

Write one coherent paragraph as the instruction. It should describe the overall layout strategy, the listed objects, and the spatial arrangement among these objects using clear relative relations such as left of, right of, in front of, behind, next to, facing, aligned with, against a wall, near a corner, or centered in. Do not include absolute distances or coordinates.

Output only the layout instruction text, no JSON or markdown.
"""
