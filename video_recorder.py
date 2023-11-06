"""A wrapper for video recording environments by rolling it out, frame by frame."""
import json
import os
import os.path
import tempfile
from typing import List, Optional

from gym import error, logger


class VideoRecorder:
    """VideoRecorder renders a nice movie of a rollout, frame by frame.
    It comes with an ``enabled`` option, so you can still use the same code on episodes where you don't want to record video.
    Note:
        You are responsible for calling :meth:`close` on a created VideoRecorder, or else you may leak an encoder process.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        metadata: Optional[dict] = None,
        base_path: Optional[str] = None,
    ):
        """Video recorder renders a nice movie of a rollout, frame by frame.
        Args:
            env (Env): Environment to take video of.
            path (Optional[str]): Path to the video file; will be randomly chosen if omitted.
            metadata (Optional[dict]): Contents to save to the metadata file.
            enabled (bool): Whether to actually record video, or just no-op (for convenience)
            base_path (Optional[str]): Alternatively, path to the video file without extension, which will be added.
        Raises:
            Error: You can pass at most one of `path` or `base_path`
            Error: Invalid path given that must have a particular file extension
        """
        try:
            # check that moviepy is now installed
            import moviepy  # noqa: F401
        except ImportError:
            raise error.DependencyNotInstalled(
                "MoviePy is not installed, run `pip install moviepy`"
            )

        self._closed = False

        self.render_history = []
       
        if path is not None and base_path is not None:
            raise error.Error("You can pass at most one of `path` or `base_path`.")

        required_ext = ".mp4"
        if path is None:
            if base_path is not None:
                # Base path given, append ext
                path = base_path + required_ext
            else:
                # Otherwise, just generate a unique filename
                with tempfile.NamedTemporaryFile(suffix=required_ext) as f:
                    path = f.name
        self.path = path

        path_base, actual_ext = os.path.splitext(self.path)

        if actual_ext != required_ext:
            raise error.Error(
                f"Invalid path given: {self.path} -- must have file extension {required_ext}."
            )

        self.frames_per_sec = 30

        self.broken = False

        # Dump metadata
        self.metadata = metadata or {}
        self.metadata["content_type"] = "video/mp4"
        self.metadata_path = f"{path_base}.meta.json"
  
        logger.info(f"Starting new video recorder writing to {self.path}")
        self.recorded_frames = []

    @property
    def functional(self):
        """Returns if the video recorder is functional, is enabled and not broken."""
        return not self.broken

    def capture_frame(self,frame):
        """Render the given `env` and add the resulting frame to the video."""
        if isinstance(frame, List):
            self.render_history += frame
            frame = frame[-1]

        if not self.functional:
            return
        if self._closed:
            logger.warn(
                "The video recorder has been closed and no frames will be captured anymore."
            )
            return
        logger.debug("Capturing video frame: path=%s", self.path)

        if frame is None:
            # Indicates a bug in the environment: don't want to raise
            # an error here.
            logger.warn(
                "Env returned None on `render()`. Disabling further rendering for video recorder by marking as "
                f"disabled: path={self.path} metadata_path={self.metadata_path}"
            )
            self.broken = True
        else:
            self.recorded_frames.append(frame)

    def close(self):
        """Flush all data to disk and close any open frame encoders."""
        if self._closed:
            return

        # Close the encoder
        if len(self.recorded_frames) > 0:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError:
                raise error.DependencyNotInstalled(
                    "MoviePy is not installed, run `pip install moviepy`"
                )

            logger.debug(f"Closing video encoder: path={self.path}")
            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            clip.write_videofile(self.path)

        # Stop tracking this for autoclose
        self._closed = True

