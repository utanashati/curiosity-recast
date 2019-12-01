# Taken from https://gitlab.crowdai.org/vizdoom/vizdoom-subcontractor/blob/master/generate_video.py
#!/usr/bin/env python3

import os, cv2
import vizdoom as vzd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from argparse import ArgumentParser
from tqdm import tqdm
import warnings

# ####################################################################################
# Hardcoded settings
# ####################################################################################

# Colors are in BGR
#                   text color, stroke color, shadow color
WHTE_TEXT_COLOR = [(255, 255, 255), None, None]
RED_TEXT_COLOR = [(0, 0, 203), (0, 0, 73), (46, 46, 46)]
GREEN_TEXT_COLOR = [(86, 206, 95), (25, 46, 34), (46, 46, 46)]
GRAY_TEXT_COLOR = [(183, 183, 183), (79, 79, 79), (46, 46, 46)]
ORANGE_TEXT_COLOR = [(15, 115, 223), (7, 35, 83), (46, 46, 46)]
AWESOME_FONT_FILENAME = "fontawesome.ttf"
DOOM_FONT_FILENAME = "DooM.ttf"

# number will be added and extension (XXXXXX.png)
FRAME_NAME_PREFIX = 'frame'
# VIDEO_CODEC = 'XVID'
VIDEO_CODEC = 'H264'
# VIDEO_CODEC = 'MJPG'


# ####################################################################################
# Functions
# ####################################################################################

def draw_text(draw, text_x, text_y, text, color, font):
    # shadow
    if shadow and color[2] is not None:
        draw.text((text_x + shadow_pos, text_y + shadow_pos), text, color[2], font=font)

    # stroke
    if text_stroke and color[1] is not None:
        draw.text((text_x + text_stroke_thickness, text_y + text_stroke_thickness), text, color[1], font=font)
        draw.text((text_x + text_stroke_thickness, text_y - text_stroke_thickness), text, color[1], font=font)
        draw.text((text_x - text_stroke_thickness, text_y + text_stroke_thickness), text, color[1], font=font)
        draw.text((text_x - text_stroke_thickness, text_y - text_stroke_thickness), text, color[1], font=font)

    draw.text((text_x, text_y), text, color[0], font=font)


def add_captions2frame(frame, text_x_pos, text_y_pos):
    pil_screen = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_screen)

    text_size = draw.textsize(title, font=upper_title_font)
    line_height = text_size[1]

    # Draw text
    draw_text(draw, text_x_pos, text_y_pos, title, RED_TEXT_COLOR, upper_title_font)
    draw_text(draw, text_x_pos, text_y_pos + 1.5 * line_height, subtitle, RED_TEXT_COLOR, title_font)

    # Put back to frame
    return np.array(pil_screen)


if __name__ == "__main__":
    parser = ArgumentParser("Creates video file from ViZDoom's recording")
    # parser.add_argument("--recording_file", "-r")
    parser.add_argument("--run_dir", "-r")
    parser.add_argument(
        "--config_file", "-c", default="gym/vizdoomgym/"
        "vizdoomgym/envs/scenarios/my_way_home_dense.cfg")
    parser.add_argument("--player", "-p", type=int, default=1)
    parser.add_argument("--title", "-t", type=str, default="title")
    parser.add_argument("--subtitle", "-s", type=str, default="subtitle")
    # parser.add_argument("--output_dir", "-o", default=None)
    parser.add_argument("--frames_output", default="frames")
    parser.add_argument("--resources_dir", default="resources")
    parser.add_argument("--fps", "-fps", default=20, type=int)
    parser.add_argument("--resolution", default="RES_640X360")
    parser.add_argument("--max_frames", default=None, type=int)
    parser.add_argument("--show_progress_bar", action="store_true", default=False)

    args = parser.parse_args()

    recordings_dir = os.path.join(args.run_dir, 'recordings')
    output_dir = os.path.join(args.run_dir, 'videos')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    recording_files = [
        f for f in os.listdir(recordings_dir)
        if os.path.isfile(os.path.join(recordings_dir, f))]

    output_files = ['_'.join(['video'] + f.split('_')[1:])[:-3] + 'mp4' for f in recording_files]
    print(output_files)

    print(recording_files)

    awesome_font_file = os.path.join(args.resources_dir, AWESOME_FONT_FILENAME)
    doom_font_file = os.path.join(args.resources_dir, DOOM_FONT_FILENAME)

    resolution = None
    try:
        resolution = getattr(vzd.ScreenResolution, args.resolution)
    except Exception:
        print("Probably not supported resolution: {}. Aborting.".format(args.resolution))
        exit(1)

    save_as_images = False
    save_as_video = True

    render_captions = True
    render_rewards = False
    render_title = True
    render_stats = True

    title = args.title
    subtitle = args.subtitle
    frames_dir = os.path.join(args.frames_output, args.resolution)

    # ####################################################################################
    # Configuration and game setup
    # ####################################################################################

    game = vzd.DoomGame()
    game.load_config(args.config_file)
    game.set_screen_resolution(resolution)
    game.set_screen_format(vzd.ScreenFormat.BGR24)
    game.set_window_visible(False)
    game.set_console_enabled(False)
    game.set_sound_enabled(False)
    game.add_game_args("+vid_forcesurface 1")
    game.add_game_args("-host 1")
    game.init()

    for recording_file, output_file in zip(recording_files, output_files):
        video_name = os.path.join(output_dir, output_file)
        game.replay_episode(os.path.join(recordings_dir, recording_file), args.player)

        screen_channels = game.get_screen_channels()
        frame_width = game.get_screen_width()
        frame_height = game.get_screen_height()
        if frame_width / frame_height != 16 / 9:
            warnings.warn("Resolution: '{}' has different aspect ration than FULL HD".format(args.resolution))
        fps = args.fps

        ratio = frame_width / 1920
        base_font_size = 30 * ratio
        text_x_pos = int(20 * ratio)
        text_y_pos = int(10 * ratio)

        # Title rendering
        upper_title_font_size = int(1.5 * base_font_size)
        title_font_size = int(2 * base_font_size)

        render_text = True
        text_stroke = True
        text_stroke_thickness = int(4 * ratio)

        shadow = True
        shadow_pos = int(8 * ratio)

        if save_as_images:
            os.makedirs(frames_dir, exist_ok=True)

        video_capture = None
        video_writer = None
        if save_as_video:
            video_capture = cv2.VideoCapture(0)
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
            video_writer = cv2.VideoWriter(video_name, fourcc, fps, (frame_width, frame_height))

        fontawesome = ImageFont.truetype(awesome_font_file, int(1.1 * base_font_size))
        font = ImageFont.truetype(doom_font_file, int(base_font_size))
        small_font = ImageFont.truetype(doom_font_file, int(0.8 * base_font_size))
        large_font = ImageFont.truetype(doom_font_file, int(1.2 * base_font_size))
        upper_title_font = ImageFont.truetype(doom_font_file, upper_title_font_size)
        title_font = ImageFont.truetype(doom_font_file, title_font_size)

        # ####################################################################################
        # Main loop
        # ####################################################################################

        frame_count = 0
        progress_bar = None
        if args.show_progress_bar:
            progress_bar = tqdm(total=args.max_frames)

        while not game.is_episode_finished():
            state = game.get_state()
            frame = state.screen_buffer
            if render_text:
                frame = add_captions2frame(frame, text_x_pos, text_y_pos)
            game.advance_action()

            if save_as_images:
                file_path = "{}/{}_{:06d}.png".format(frames_dir, FRAME_NAME_PREFIX, frame_count)
                cv2.imwrite(file_path, frame)

            if save_as_video:
                video_writer.write(frame)

            frame_count += 1
            if args.show_progress_bar:
                progress_bar.update(1)
            else:
                print("Saved frame {:06d}".format(frame_count))

            if args.max_frames is not None and frame_count > int(args.max_frames):
                break

        if args.show_progress_bar:
            progress_bar.close()
        if save_as_video:
            video_capture.release()
            video_writer.release()
