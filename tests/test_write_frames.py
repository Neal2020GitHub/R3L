"""Tests for utils.r3l.plot.write_frames — the GIF/MP4 assembler.

Locks in the imageio duration-unit fix: the old `_write_gif(duration=0.25)` wrote
0.25 *milliseconds* per frame (truncated to 0), so GIFs played at the viewer's
minimum-delay clamp instead of the intended rate. With `fps=`, every frame must
carry ~1000/fps ms and the GIF must loop forever.
"""
import os
import shutil
import tempfile
import unittest

import imageio
import numpy as np
from PIL import Image

from utils.r3l.plot import write_frames


class TestWriteFrames(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.frame_paths = []
        for i in range(4):
            arr = np.full((64, 64, 3), (i * 60) % 256, dtype=np.uint8)
            p = os.path.join(self.tmp, f"frame_{i:04d}.png")
            Image.fromarray(arr).save(p)
            self.frame_paths.append(p)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_gif_per_frame_delay_and_loop(self):
        gif = os.path.join(self.tmp, "out.gif")
        write_frames(self.frame_paths, gif, fps=24, fmt="gif")
        self.assertTrue(os.path.exists(gif))

        im = Image.open(gif)
        durs = []
        try:
            i = 0
            while True:
                im.seek(i)
                durs.append(im.info.get("duration"))
                i += 1
        except EOFError:
            pass
        real = [d for d in durs if d is not None]
        self.assertGreaterEqual(len(real), 1)
        for d in real:
            self.assertAlmostEqual(d, 1000 / 24, delta=2)  # ~41.7 ms, not 0
        self.assertEqual(im.info.get("loop"), 0)  # loop forever

    def test_mp4_written_and_readable(self):
        mp4 = os.path.join(self.tmp, "out.mp4")
        write_frames(self.frame_paths, mp4, fps=24, fmt="mp4")
        self.assertTrue(os.path.exists(mp4))
        self.assertGreater(os.path.getsize(mp4), 0)

        reader = imageio.get_reader(mp4)
        try:
            count = sum(1 for _ in reader)  # ffmpeg backend reports length=inf; count by iteration
        finally:
            reader.close()
        self.assertEqual(count, len(self.frame_paths))

    def test_unsupported_format_raises(self):
        with self.assertRaises(ValueError):
            write_frames(self.frame_paths, os.path.join(self.tmp, "x.webm"),
                         fps=24, fmt="webm")


if __name__ == "__main__":
    unittest.main()