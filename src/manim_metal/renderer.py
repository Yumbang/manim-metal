"""MetalRenderer — duck-typed to match CairoRenderer's interface."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
from manim import config, logger
from manim.scene.scene_file_writer import SceneFileWriter
from manim.utils.exceptions import EndSceneEarlyException
from manim.utils.hashing import get_hash_from_play_call
from manim.utils.iterables import list_update

from manim_metal.metal_camera import MetalCamera

if TYPE_CHECKING:
    from manim.animation.animation import Animation
    from manim.mobject.mobject import Mobject, _AnimationBuilder
    from manim.scene.scene import Scene
    from manim.typing import PixelArray


class MetalRenderer:
    """A renderer using Apple's Metal API.

    Duck-typed to match :class:`~manim.renderer.cairo_renderer.CairoRenderer`.

    Attributes
    ----------
    num_plays : int
        Number of play() calls in the scene.
    time : float
        Time elapsed since scene initialisation.
    """

    def __init__(
        self,
        file_writer_class: type[SceneFileWriter] = SceneFileWriter,
        skip_animations: bool = False,
        **kwargs: Any,
    ) -> None:
        self._file_writer_class = file_writer_class
        self.camera = MetalCamera()
        self._original_skipping_status = skip_animations
        self.skip_animations = skip_animations
        self.animations_hashes: list[str | None] = []
        self.num_plays = 0
        self.time = 0.0
        self.static_image: PixelArray | None = None

    def init_scene(self, scene: Scene) -> None:
        """Create the file writer for this scene."""
        self.file_writer: Any = self._file_writer_class(
            self,
            scene.__class__.__name__,
        )

    # ------------------------------------------------------------------
    # play() — orchestrates a single animation
    # ------------------------------------------------------------------

    def play(
        self,
        scene: Scene,
        *args: Animation | Mobject | _AnimationBuilder,
        **kwargs: Any,
    ) -> None:
        # Reset skip_animations to original state
        self.skip_animations = self._original_skipping_status
        self.update_skipping_status()

        scene.compile_animation_data(*args, **kwargs)

        if self.skip_animations:
            logger.debug(f"Skipping animation {self.num_plays}")
            hash_current_animation = None
            self.time += scene.duration
        else:
            if config["disable_caching"]:
                logger.info("Caching disabled.")
                hash_current_animation = f"uncached_{self.num_plays:05}"
            else:
                assert scene.animations is not None
                hash_current_animation = get_hash_from_play_call(
                    scene,
                    self.camera,
                    scene.animations,
                    scene.mobjects,
                )
                if self.file_writer.is_already_cached(hash_current_animation):
                    logger.info(
                        "Animation %d : Using cached data (hash : %s)",
                        self.num_plays,
                        hash_current_animation,
                    )
                    self.skip_animations = True
                    self.time += scene.duration

        self.file_writer.add_partial_movie_file(hash_current_animation)
        self.animations_hashes.append(hash_current_animation)
        logger.debug(
            "List of the first few animation hashes of the scene: %(h)s",
            {"h": str(self.animations_hashes[:5])},
        )

        self.file_writer.begin_animation(not self.skip_animations)
        scene.begin_animations()

        # Save static image to avoid re-rendering non-moving objects
        self.save_static_frame_data(scene, scene.static_mobjects)

        if scene.is_current_animation_frozen_frame():
            self.update_frame(scene, mobjects=scene.moving_mobjects)
            self.freeze_current_frame(scene.duration)
        else:
            scene.play_internal()

        self.file_writer.end_animation(not self.skip_animations)
        self.num_plays += 1

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------

    def update_frame(
        self,
        scene: Scene,
        mobjects: Iterable[Mobject] | None = None,
        include_submobjects: bool = True,
        ignore_skipping: bool = True,
        **kwargs: Any,
    ) -> None:
        """Update the frame by rendering mobjects through Metal."""
        if self.skip_animations and not ignore_skipping:
            return
        if not mobjects:
            mobjects = list_update(scene.mobjects, scene.foreground_mobjects)
        if self.static_image is not None:
            self.camera.set_frame_to_background(self.static_image)
        else:
            self.camera.reset()

        kwargs["include_submobjects"] = include_submobjects
        self.camera.capture_mobjects(mobjects, **kwargs)

    def render(
        self,
        scene: Scene,
        time: float,
        moving_mobjects: Iterable[Mobject] | None = None,
    ) -> None:
        """Called every frame from scene.play_internal()."""
        self.update_frame(scene, moving_mobjects)
        self.add_frame(self.get_frame())

    def get_frame(self) -> PixelArray:
        """Get the current frame as NumPy array (height, width, 4) RGBA uint8."""
        return np.array(self.camera.pixel_array)

    def add_frame(self, frame: PixelArray, num_frames: int = 1) -> None:
        """Add frame(s) to the video file."""
        dt = 1 / self.camera.frame_rate
        if self.skip_animations:
            return
        self.time += num_frames * dt
        self.file_writer.write_frame(frame, num_frames=num_frames)

    def freeze_current_frame(self, duration: float) -> None:
        """Add the current frame as a static hold for the given duration."""
        dt = 1 / self.camera.frame_rate
        self.add_frame(self.get_frame(), num_frames=int(duration / dt))

    def show_frame(self, scene: Scene) -> None:
        """Open the current frame in the default image viewer."""
        self.update_frame(scene, ignore_skipping=True)
        self.camera.get_image().show()

    def save_static_frame_data(
        self,
        scene: Scene,
        static_mobjects: Iterable[Mobject],
    ) -> PixelArray | None:
        """Render and cache static mobjects to avoid re-rendering them each frame."""
        self.static_image = None
        if not static_mobjects:
            return None
        self.update_frame(scene, mobjects=static_mobjects)
        self.static_image = self.get_frame()
        return self.static_image

    # ------------------------------------------------------------------
    # Skipping / scene lifecycle
    # ------------------------------------------------------------------

    def update_skipping_status(self) -> None:
        """Check whether the current animation should be skipped."""
        if self.file_writer.sections[-1].skip_animations:
            self.skip_animations = True
        if config["save_last_frame"]:
            self.skip_animations = True
        if config.from_animation_number > 0 and self.num_plays < config.from_animation_number:
            self.skip_animations = True
        if config.upto_animation_number >= 0 and self.num_plays > config.upto_animation_number:
            self.skip_animations = True
            raise EndSceneEarlyException()

    def scene_finished(self, scene: Scene) -> None:
        """Called when the scene is done rendering."""
        if self.num_plays:
            self.file_writer.finish()
        elif config.write_to_movie:
            config.save_last_frame = True
            config.write_to_movie = False
        else:
            self.static_image = None
            self.update_frame(scene)

        if config["save_last_frame"]:
            self.static_image = None
            self.update_frame(scene)
            self.file_writer.save_image(self.camera.get_image())
